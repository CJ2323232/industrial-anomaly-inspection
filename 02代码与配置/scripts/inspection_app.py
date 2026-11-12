"""Chinese offline desktop entry: select -> inspect -> review -> save."""
import argparse
import os
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
if __package__:
    from .inspection_runtime import ROOT, DATA, OUTPUT, InputGuard, Inspector, read_json, DECISION
else:
    from inspection_runtime import ROOT, DATA, OUTPUT, InputGuard, Inspector, read_json, DECISION

class InspectionApp:
    def __init__(self, root):
        self.root=root; self.guard=InputGuard(); self.engine=None; self.events=queue.Queue()
        self.selected=None; self.output=None; self.busy=False
        self.policy=read_json(DECISION)
        self.threshold_text=f'整图阈值 {self.policy["image_threshold"]:.6f}  |  像素阈值 {self.policy["provisional_pixel_threshold"]:.6f}（开发候选）'
        root.title('工业视觉异常检测 · WRN50开发候选')
        root.geometry('1180x850'); root.minsize(900,680); root.configure(bg='#f3f6fa')
        style=ttk.Style(root); style.theme_use('clam')
        style.configure('TFrame',background='#f3f6fa')
        style.configure('TLabel',background='#f3f6fa',font=('Microsoft YaHei UI',10))
        style.configure('TButton',font=('Microsoft YaHei UI',10),padding=8)
        frame=ttk.Frame(root,padding=20);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='金属螺母 · 异常检测与定位',font=('Microsoft YaHei UI',20,'bold')).pack(anchor='w')
        ttk.Label(frame,text='PatchCore + WideResNet50  |  256 × 256  |  sigma 4  |  加载已有模型，无需训练').pack(anchor='w',pady=(5,12))
        actions=ttk.Frame(frame);actions.pack(fill='x')
        self.choose_btn=ttk.Button(actions,text='① 选择图片',command=self.choose);self.choose_btn.pack(side='left',padx=(0,8))
        examples=[p for p in ['metal_nut/test/scratch/000.png','metal_nut/test/color/001.png','metal_nut/test/good/005.png'] if p in self.guard.manifest['splits']['development']]
        self.examples=ttk.Combobox(actions,values=examples,state='readonly',width=35)
        self.examples.pack(side='left',padx=(0,8));self.examples.current(0)
        self.sample_btn=ttk.Button(actions,text='载入开发集样例',command=lambda:self.select(DATA/self.examples.get()))
        self.sample_btn.pack(side='left',padx=(0,8))
        self.run_btn=ttk.Button(actions,text='② 开始检测',command=self.run,state='disabled');self.run_btn.pack(side='left')
        self.path_label=ttk.Label(frame,text='请选择图片，或载入一个开发集样例。',wraplength=1080);self.path_label.pack(anchor='w',pady=10)
        self.status=ttk.Label(frame,text='等待选择图片',font=('Microsoft YaHei UI',12,'bold'));self.status.pack(anchor='w')
        self.progress=ttk.Progressbar(frame,mode='indeterminate');self.progress.pack(fill='x',pady=8)
        self.metrics=ttk.Label(frame,text=self.threshold_text,wraplength=1080);self.metrics.pack(anchor='w',pady=(0,10))
        self.notebook=ttk.Notebook(frame)
        self.panels={}
        for name in ['原图','缺陷叠加','热力图','预测掩码']:
            pane=ttk.Frame(self.notebook);self.notebook.add(pane,text=name)
            label=ttk.Label(pane,text='检测后显示',anchor='center');label.pack(fill='both',expand=True,padx=10,pady=10)
            self.panels[name]=label
        bottom=ttk.Frame(frame);bottom.pack(side='bottom',fill='x',pady=(12,8))
        self.folder_btn=ttk.Button(bottom,text='③ 打开结果文件夹',command=self.open_folder,state='disabled');self.folder_btn.pack(side='left')
        self.report_btn=ttk.Button(bottom,text='打开完整报告',command=self.open_report,state='disabled');self.report_btn.pack(side='left',padx=8)
        ttk.Label(frame,text='输出自动保存到项目 / 04实验结果 / 14单图检测。整图判定与像素候选区域分开显示。',wraplength=1080).pack(side='bottom',anchor='w')
        ttk.Label(frame,text='当前模型适用于本项目金属螺母；像素阈值尚未独立验证。保留测试图不通过此入口打开。',wraplength=1080).pack(side='bottom',anchor='w')
        self.notebook.pack(fill='both',expand=True)
        root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def display(self,name,image):
        self.root.update_idletasks()
        size=(max(100,self.notebook.winfo_width()-50),max(100,self.notebook.winfo_height()-65))
        im=image.copy();im.thumbnail((min(1000,size[0]),min(490,size[1])),Image.Resampling.LANCZOS)
        photo=ImageTk.PhotoImage(im);self.panels[name].configure(image=photo,text='');self.panels[name].image=photo

    def choose(self):
        folder=ROOT/'03原始数据/手动检测输入';folder.mkdir(exist_ok=True)
        path=filedialog.askopenfilename(parent=self.root,title='选择一张金属螺母图片',initialdir=folder,filetypes=[('图片','*.png *.jpg *.jpeg *.bmp *.tif *.tiff')])
        if path:self.select(Path(path))

    def select(self,path):
        try:image,identity=self.guard.load(path)
        except Exception as exc:messagebox.showerror('不能载入图片',str(exc),parent=self.root);return
        self.selected=Path(path);self.output=None
        self.folder_btn.configure(state='disabled');self.report_btn.configure(state='disabled')
        for panel in self.panels.values():panel.configure(image='',text='检测后显示');panel.image=None
        self.display('原图',image);self.notebook.select(0)
        self.path_label.configure(text=str(path));self.status.configure(text='图片已载入，点击“开始检测”')
        self.metrics.configure(text=f'原图 {identity["width"]} × {identity["height"]}  |  '+self.threshold_text)
        self.run_btn.configure(state='normal')

    def run(self):
        if self.busy or self.selected is None:return
        self.busy=True;self.output=None
        for b in (self.choose_btn,self.sample_btn,self.run_btn,self.folder_btn,self.report_btn):b.configure(state='disabled')
        self.examples.configure(state='disabled');self.progress.start(12)
        self.status.configure(text='正在加载模型并检测…首次运行需要稍等')
        selected=self.selected
        def work():
            try:
                if self.engine is None:self.engine=Inspector()
                self.events.put(('ok',self.engine.infer(selected)))
            except Exception as exc:
                log=ROOT/'.cache/inspection';log.mkdir(parents=True,exist_ok=True)
                (log/'last_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                self.engine=None;self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:kind,payload=self.events.get_nowait()
        except queue.Empty:pass
        else:
            self.busy=False;self.progress.stop();self.examples.configure(state='readonly')
            for b in (self.choose_btn,self.sample_btn,self.run_btn):b.configure(state='normal')
            if kind=='error':
                self.status.configure(text='检测失败，请查看错误提示')
                messagebox.showerror('检测失败',payload,parent=self.root)
            else:
                self.output,r=payload
                self.status.configure(text=('检测为异常' if r['image_is_anomalous'] else '整图未判异常')+' · 结果已保存')
                self.metrics.configure(text=f'整图分数 {r["raw_image_score"]:.6f} / 阈值 {r["image_threshold"]:.6f}    |    候选区域 {r["predicted_area_fraction"]:.2%}\n本次检测 {r["timing"]["decode_preprocess_transfer_infer_restore_ms"]:.1f} ms（不含模型加载和导出）  |  {r["model"]["device"]}')
                for name in self.panels:
                    with Image.open(self.output/(name+'.png')) as im:self.display(name,im)
                self.notebook.select(1);self.folder_btn.configure(state='normal');self.report_btn.configure(state='normal')
        self.root.after(100,self.poll)

    def open_folder(self):
        if self.output:os.startfile(str(self.output))

    def open_report(self):
        if self.output:os.startfile(str(self.output/'检测报告.html'))

    def close(self):
        if self.busy:
            messagebox.showinfo('正在检测','请等本次结果保存后再关闭窗口。',parent=self.root);return
        self.root.destroy()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,help='命令行单图检测，不打开窗口')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0'])
    parser.add_argument('--preflight',action='store_true',help='检查配置、模型哈希与Tk，不运行推理')
    args=parser.parse_args()
    if args.preflight:
        Inspector(args.device)
        root=tk.Tk();root.withdraw();root.update();root.destroy()
        print('PREFLIGHT_OK: model hash, policy, split manifest and Tk; no image inference.');return
    if args.image:
        output,result=Inspector(args.device).infer(args.image)
        print(output);print('image_anomaly=',result['image_is_anomalous'],'score=',result['raw_image_score']);return
    root=tk.Tk();InspectionApp(root);root.mainloop()

if __name__=='__main__':main()
