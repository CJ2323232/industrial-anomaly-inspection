"""Local versioned Tkinter demo; inference only, no fitting."""
from pathlib import Path
import argparse,json,os,queue,threading,traceback,webbrowser
from datetime import datetime
import tkinter as tk
from tkinter import filedialog,ttk
from PIL import Image,ImageTk,UnidentifiedImageError
from delivery_runtime import DeliveryInspector,ROOT,PROFILES,read_json

def main():
    profiles=read_json(PROFILES)['profiles']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=list(profiles),default='r18_reference')
    parser.add_argument('--initial-dir',type=Path,default=ROOT/'03原始数据/手动检测输入')
    args=parser.parse_args()
    win=tk.Tk();win.title('工业异常检测 · 最终演示');win.geometry('1120x850')
    labels={v['name']:k for k,v in profiles.items()}
    selected=tk.StringVar(value=profiles[args.profile]['name'])
    row=ttk.Frame(win);row.pack(fill='x',padx=12,pady=12)
    ttk.Label(row,text='版本：').pack(side='left')
    combo=ttk.Combobox(row,textvariable=selected,values=list(labels),state='readonly',width=32)
    combo.pack(side='left')
    ttk.Label(win,text='R18为默认参考；WRN50 A为定位对照；WRN50 C可试用，尚缺新增同域独立验证。\n仅适用于原metal_nut任务；整图判断与候选像素区域分别显示。').pack()
    status=tk.StringVar(value='请选择开发集图片或自己的新图片。历史70张测试图拒绝访问。')
    ttk.Label(win,textvariable=status,wraplength=1080).pack(padx=12,pady=10)
    tabs=ttk.Notebook(win);tabs.pack(fill='both',expand=True,padx=12,pady=8)
    panels={}
    for name in ['原图','热力图','缺陷叠加','预测掩码']:
        panel=ttk.Label(tabs);tabs.add(panel,text=name);panels[name]=panel
    events=queue.Queue();last=[None];busy=[False];cache={}
    logdir=ROOT/'04实验结果/26最终演示'
    def log(event,**details):
        logdir.mkdir(parents=True,exist_ok=True)
        record=dict(time=datetime.now().astimezone().isoformat(),event=event,**details)
        with (logdir/'gui_events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    def reset_output():
        last[0]=None;report_button.config(state='disabled');folder_button.config(state='disabled')
        for panel in panels.values():panel.config(image='');panel.image=None
    def change_profile(event=None):
        reset_output();status.set('已选择 '+selected.get()+'，请选择图片进行检测。')
        log('profile_selected',profile=labels[selected.get()])
    def open_result(report):
        if last[0] is None:return
        out=last[0];result=read_json(out/'结果.json')
        view=tk.Toplevel(win);view.title('检测报告' if report else '结果文件');view.geometry('820x560')
        body=tk.Text(view,wrap='word',font=('Microsoft YaHei',11));body.pack(fill='both',expand=True,padx=12,pady=12)
        if report:
            lines=[result['profile_name'],result.get('evidence_limit') or '参考演示版本',
                '图片：'+(result['input'].get('manifest_image') or Path(result['input']['selected_path']).name),
                '整图结果：'+('异常' if result['image_is_anomalous'] else '整图未判异常'),
                '原始图像分数：'+str(result['raw_image_score']),
                '图像阈值：'+str(result['image_threshold']),
                '像素阈值：'+str(result['pixel_threshold']),
                '像素候选面积：'+format(result['predicted_area_fraction'],'.2%'),
                '评分：'+result['scoring'],
                '整图判定与像素掩码独立；分数不是概率。',
                '完整HTML与图像文件已保存在：',str(out)]
        else:
            lines=['结果保存目录：',str(out),'',*[(x.name+'    '+str(x.stat().st_size)+' bytes') for x in sorted(out.iterdir()) if x.is_file()]]
        body.insert('1.0',(chr(10)*2).join(lines));body.config(state='disabled')
        def system_open():
            target=out/'检测报告.html' if report else out
            try:
                if os.name=='nt':os.startfile(str(target))
                elif not webbrowser.open(target.as_uri()):raise OSError('系统未能打开该路径。')
                log('system_open_requested',kind='report' if report else 'folder',path=str(target))
            except Exception as e:
                status.set('系统应用未能打开结果：'+str(e)+'；仍可在本窗口查看报告与文件清单。')
                log('system_open_failed',message=str(e))
        ttk.Button(view,text='用系统应用打开',command=system_open).pack(side='left',padx=12,pady=8)
        ttk.Button(view,text='关闭',command=view.destroy).pack(side='right',padx=12,pady=8)
        log('report_view_opened' if report else 'files_view_opened',output=str(out))
    def select_image():
        if busy[0]:return
        path=filedialog.askopenfilename(parent=win,
            initialdir=str(args.initial_dir if args.initial_dir.is_dir() else ROOT),
            filetypes=[('Images','*.png *.jpg *.jpeg *.bmp *.tif *.tiff')])
        if not path:log('file_selection_cancelled');return
        reset_output();busy[0]=True;key=labels[selected.get()]
        choose_button.config(state='disabled');combo.config(state='disabled')
        status.set('正在检测；首次加载模型会较慢。')
        log('inference_requested',profile=key,input=path)
        def worker():
            try:
                if key not in cache:cache[key]=DeliveryInspector(key)
                events.put(('ok',cache[key].infer(path)))
            except Exception as e:
                message='文件不是可解码的图片或已经损坏，请选择有效PNG/JPG等图片。' if isinstance(e,UnidentifiedImageError) else str(e)
                events.put(('error',(message,traceback.format_exc())))
        threading.Thread(target=worker,daemon=True).start()
    choose_button=ttk.Button(row,text='选择图片并检测',command=select_image)
    choose_button.pack(side='left',padx=12)
    report_button=ttk.Button(row,text='查看结果报告',command=lambda:open_result(True),state='disabled')
    report_button.pack(side='left')
    folder_button=ttk.Button(row,text='查看结果文件',command=lambda:open_result(False),state='disabled')
    folder_button.pack(side='left')
    combo.bind('<<ComboboxSelected>>',change_profile)
    def poll():
        try:
            kind,value=events.get_nowait();busy[0]=False
            choose_button.config(state='normal');combo.config(state='readonly')
            if kind=='error':
                logdir.mkdir(parents=True,exist_ok=True)
                (logdir/'last_error.log').write_text(value[1],encoding='utf-8')
                status.set('检测失败或输入被拒绝：'+value[0])
                log('input_rejected',message=value[0])
                # Errors stay inline; do not interrupt the desktop.
            else:
                out,result=value;last[0]=out
                report_button.config(state='normal');folder_button.config(state='normal')
                status.set(f"{result['profile_name']} | {'异常' if result['image_is_anomalous'] else '整图未判异常'} | 原始分数 {result['raw_image_score']:.6f} / 阈值 {result['image_threshold']:.6f} | 像素候选区域 {result['predicted_area_fraction']:.2%}")
                for name,panel in panels.items():
                    with Image.open(out/(name+'.png')) as im:
                        im=im.copy();im.thumbnail((950,610));photo=ImageTk.PhotoImage(im)
                    panel.config(image=photo);panel.image=photo
                tabs.select(0);log('result_displayed',profile=result['profile'],output=str(out))
        except queue.Empty:pass
        win.after(150,poll)
    def close():
        if busy[0]:
            log('close_blocked_while_busy')
            status.set('正在检测，请等当前结果保存完成再关闭；不会自动重复检测。')
        else:log('closed');win.destroy()
    win.protocol('WM_DELETE_WINDOW',close)
    log('opened',initial_profile=args.profile)
    poll();win.mainloop()
if __name__=='__main__':main()
