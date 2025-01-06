from django import template
import os

register = template.Library()


# Code dùng để lấy file name, ví dụ Nếu value = "/folder/subfolder/file.txt" thì kết quả sẽ trả về "file.txt"
@register.filter
def basename(value):
    return os.path.basename(value)