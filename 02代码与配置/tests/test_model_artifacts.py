"""Small byte fixtures only: never deserialize a model or read local weights."""
from pathlib import Path
import hashlib,json,tempfile,unittest
from scripts.model_artifacts import ArtifactError,inspect_folder,import_bundle,export_bundle,resolve_artifact

class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.original=self.root/"old";self.original.mkdir()
        (self.original/"model.pt").write_bytes(b"synthetic bytes, not a torch checkpoint")
        (self.original/"config.json").write_text('{"backbone":"fixture","image_size":[2,2]}',encoding="utf-8")
        self.entry=dict(artifact_id="fixture-v1",profiles=["fixture"],legacy_checkpoint="old/model.pt",
            files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.original.iterdir()})
        self.catalog=dict(artifacts={"fixture-v1":self.entry})
        self.addCleanup(self.temp.cleanup)
    def export(self):
        return export_bundle("fixture",self.root/"bundle",root=self.root,catalog=self.catalog)
    def test_export_import_idempotence_and_relocation(self):
        source=self.export();store=self.root/"new-location"
        dest,created=import_bundle(source,store,self.catalog)
        self.assertTrue(created)
        self.assertFalse(import_bundle(source,store,self.catalog)[1])
        path,config,origin=resolve_artifact("fixture",root=self.root/"no-legacy",store=store,catalog=self.catalog)
        self.assertEqual(path,dest/"model.pt");self.assertEqual(origin,"imported")
        self.assertEqual(config["backbone"],"fixture")
    def test_bad_sha_rejected_before_destination_created(self):
        source=self.export();f=source/"model.pt";f.write_bytes(b"x"*f.stat().st_size)
        with self.assertRaisesRegex(ArtifactError,"SHA256"):import_bundle(source,self.root/"dest",self.catalog)
        self.assertFalse((self.root/"dest").exists())
    def test_modified_manifest_cannot_authorize_different_bytes(self):
        source=self.export();m=json.loads((source/"bundle.json").read_text())
        m["files"]["model.pt"]["sha256"]="0"*64
        (source/"bundle.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ArtifactError,"清单"):import_bundle(source,self.root/"dest",self.catalog)
    def test_extra_file_and_missing_config_are_diagnosed(self):
        source=self.export();(source/"unexpected.txt").write_text("unregistered")
        self.assertTrue(any("未登记" in x for x in inspect_folder(source,self.entry,True)))
        (source/"config.json").unlink()
        self.assertTrue(any("config.json" in x for x in inspect_folder(source,self.entry,True)))
    def test_unknown_id_and_path_traversal_rejected(self):
        source=self.export();m=json.loads((source/"bundle.json").read_text());m["artifact_id"]="../escape"
        (source/"bundle.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ArtifactError,"未知"):import_bundle(source,self.root/"dest",self.catalog)
    def test_invalid_manifest_schema_is_diagnosed(self):
        source=self.export()
        for invalid in [[],None,{"artifact_id":[]},{"artifact_id":4}]:
            with self.subTest(manifest=invalid):
                (source/"bundle.json").write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ArtifactError,"JSON对象"):
                    import_bundle(source,self.root/"dest",self.catalog)
                self.assertFalse((self.root/"dest").exists())
    def test_bad_existing_import_not_overwritten_or_hidden_by_fallback(self):
        source=self.export();store=self.root/"store";dest,_=import_bundle(source,store,self.catalog)
        (dest/"model.pt").write_bytes(b"broken")
        with self.assertRaises(ArtifactError):import_bundle(source,store,self.catalog)
        with self.assertRaises(ArtifactError):resolve_artifact("fixture",root=self.root,store=store,catalog=self.catalog)
        self.assertEqual((dest/"model.pt").read_bytes(),b"broken")
    def test_missing_artifacts_explain_acquisition_and_analysis_alternative(self):
        with self.assertRaisesRegex(ArtifactError,"run_saved_score_example"):
            resolve_artifact("fixture",root=self.root/"empty",catalog=self.catalog)

if __name__=="__main__":unittest.main()

