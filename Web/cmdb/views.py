
from django.http import HttpResponse
from django.shortcuts import render
# Create your views here.
from cmdb import save_predict_img

app_name='cmdb'

def put_ajax(request):
    if request.is_ajax():
        f_obj = request.FILES.get("img")
        print(f_obj)
        name = f_obj.name
        f_write = open("static/upload/dlrb.jpg", "wb") #不用name了，采用覆盖大法简单粗暴点吧
        for line in f_obj:
            f_write.write(line)
        return HttpResponse("上传成功")
    return render(request, 'upload.html')

#调用打分并保存结果
def show(request):
    img_file = "C:/Users/miaohualin/Desktop/Web/static/upload/dlrb.jpg"
    save_path = "C:/Users/miaohualin/Desktop/Web/static/result/dlrb.jpg"
    result_list = save_predict_img(img_file=img_file, save_path=save_path)
    return render(request, "show.html",{'data':result_list})