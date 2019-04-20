
from django.http import HttpResponse
from django.shortcuts import render, redirect
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from cmdb import save_predict_img
import base64

app_name='cmdb'


@csrf_exempt
def WebCamera(request):
    if request.is_ajax():
        data = request.POST
        print(data)
        strs = data.get('base64')
        img = base64.b64decode(strs)
        f_write = open("static/upload/base64.jpg", "wb")
        f_write.write(img)
        f_write.close()
        return HttpResponse("上传成功")
    return render(request, 'WebCamera.html')


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
    img_file = "C:/Users/miaohualin/Desktop/Web/static/upload/base64.jpg"
    save_path = "C:/Users/miaohualin/Desktop/Web/static/result/base64.jpg"
    result_list = save_predict_img(img_file=img_file, save_path=save_path)
    return render(request, "show.html",{'data':result_list})