"""Model release tests use byte fixtures and an injected HTTPS response."""
from pathlib import Path
import hashlib,io,json,tempfile,unittest,zipfile
from unittest.mock import Mock
from scripts.model_artifacts import ArtifactError,export_bundle
from scripts.install_model_release import install_archive,verify_archive,download_archive,HTTPSRedirect

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);original=self.root/"old";original.mkdir()
        (original/"model.pt").write_bytes(b"small fake checkpoint")
        (original/"config.json").write_text("{}")
        self.entry=dict(artifact_id="fixture-v1",profiles=["fixture"],legacy_checkpoint="old/model.pt",
            files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in original.iterdir()})
        self.catalog=dict(artifacts={"fixture-v1":self.entry})
        bundle=export_bundle("fixture",self.root/"bundle",root=self.root,catalog=self.catalog)
        self.payload={"bundle/"+f.name:f.read_bytes() for f in bundle.iterdir()}
        self.payload["NOTICES.md"]=b"Retain the original licenses."
        self.path=self.root/"fixture.zip"
        self.make_zip(self.payload)
        self.releases=dict(artifacts={"fixture-v1":self.release})
    def make_zip(self,payload):
        with zipfile.ZipFile(self.path,"w") as z:
            for name,content in payload.items():z.writestr(name,content)
        self.release=dict(bytes=self.path.stat().st_size,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            status="prepared_not_published",url=None,files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in self.payload.items()})
    def install(self):
        return install_archive(self.path,"fixture",self.root/"store",self.releases,self.catalog)
    def test_install_preserves_notices_and_is_idempotent(self):
        result=self.install()
        self.assertTrue(result["created"]);self.assertFalse(self.install()["created"])
        self.assertEqual((Path(result["notices"])/"NOTICES.md").read_bytes(),self.payload["NOTICES.md"])
    def test_bad_archive_sha_rejected(self):
        self.path.write_bytes(b"x"*self.path.stat().st_size)
        with self.assertRaisesRegex(ArtifactError,"SHA256"):self.install()
        self.assertFalse((self.root/"store").exists())
    def test_unregistered_traversal_entry_rejected(self):
        self.make_zip({**self.payload,"../outside":b"unsafe"})
        with self.assertRaisesRegex(ArtifactError,"未登记"):verify_archive(self.path,self.release)
        self.assertFalse((self.root/"outside").exists())
    def test_duplicate_entries_rejected(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            with zipfile.ZipFile(self.path,"a") as z:z.writestr("NOTICES.md",b"duplicate")
        self.release.update(bytes=self.path.stat().st_size,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ArtifactError,"重复"):verify_archive(self.path,self.release)
    def test_inner_catalog_is_still_enforced(self):
        self.payload["bundle/model.pt"]=b"untrusted replacement"
        self.make_zip(self.payload);self.releases["artifacts"]["fixture-v1"]=self.release
        with self.assertRaises(ArtifactError):self.install()
    def test_modified_existing_notices_are_not_overwritten(self):
        r=self.install();f=Path(r["notices"])/"NOTICES.md";f.write_text("modified")
        with self.assertRaisesRegex(ArtifactError,"许可"):self.install()
        self.assertEqual(f.read_text(),"modified")
    def test_unpublished_release_does_not_contact_network(self):
        opener=Mock()
        with self.assertRaisesRegex(ArtifactError,"尚未"):download_archive(self.release,self.root/"download.zip",opener)
        opener.assert_not_called()
    def test_mock_https_download_verified_and_corruption_removed(self):
        class Response(io.BytesIO):
            def geturl(self):return "https://example.org/model.zip"
        self.release.update(status="published",url="https://example.org/model.zip")
        dest=self.root/"download.zip"
        download_archive(self.release,dest,lambda *a,**k:Response(self.path.read_bytes()))
        self.assertEqual(dest.read_bytes(),self.path.read_bytes())
        bad=self.root/"bad.zip"
        with self.assertRaises(ArtifactError):
            download_archive(self.release,bad,lambda *a,**k:Response(b"invalid"))
        self.assertFalse(bad.exists())
    def test_insecure_redirect_rejected(self):
        with self.assertRaisesRegex(ArtifactError,"HTTPS"):
            HTTPSRedirect().redirect_request(None,None,302,"",{},"http://example.org/model.zip")

if __name__=="__main__":unittest.main()

