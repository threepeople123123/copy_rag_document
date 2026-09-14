# #导包,发起请求使用urllib库的request请求模块
# import urllib.request
# # urlopen()向URL发请求,返回响应对象,注意url必须完整
# response=urllib.request.urlopen('https://www.google.com.hk/')
# print(response)
#
# html = response.read().decode('utf-8')
# #打印响应内容
# print(html)
#
#
# #向网站发送get请求
# response=urllib.request.urlopen('http://httpbin.org/get')
# html = response.read().decode()
# print(html)
#
#
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36'}
#
# print("==========================")
#
# req = urllib.request.Request(url='http://httpbin.org/get',headers=headers)
# user_agent = urllib.request.urlopen(req)
# print(user_agent.read().decode('utf-8'))
#
# print("==========================")
#
# from urllib import parse
#
# baidu_url = ("https://www.baidu.com/s?ie=utf-8&csq=1&pstg=20&mod=2&newMode=1&isbd=1&cqid=c67c64240036224d&istc=1284&ver=Rd6sTwMHOO4ajeqho_TYmO9Y00d6ZCiQCYK&chk=6aa644ee&isid=c28caa5500361f1a&f4s=1&_ck=157775.0.-1.-1.-1.-1.-1&ie=utf-8&f=3&rsv_bp=1&tn=baidu&wd=%E7%88%AC%E8%99%AB%E7%9A%84%E7%A7%8D%E7%B1%BB&oq=%25E7%2588%25AC%25E8%2599%25AB&rsv_pq=c28caa5500361f1a&rsv_t=f233%2FK0PcdqzmwmTXHTxrErO7lQUtYjeV4mT%2BE7NyJuXLsfUXaVwhjIoxA8b&rqlang=cn&rsv_dl=ts_0&rsv_btype=t&rsv_sug1=4&rsv_sug7=100&rsv_sug3=3&rsv_sug2=1&prefixsug=%25E7%2588%25AC%25E8%2599%25AB%25E7%259A%2584&rsp=0&inputT=12168&rsv_sug4=12608&bs=%E7%88%AC%E8%99%AB&rsv_isid=72949_73009_73053_73319_73290_73375_73584_73682_73788_73792_73799_73867_73902_73893_73909_73947_73951_73962_73963_74000_74015_73976_74055_74061_74141_74177_74374_74262_74257_74321_74346_74193_74215_74297_74464_74368_74394_74403_74387_74425_74407_74500_74508_74515_74526_74536_74530_74544_74590_74601_74644_74626_74668_74666_74670_74683_74759_74711_74722_74673_74794_74803_74809_74825_74835_74839_74838&isnop=1")
#
# query_string = parse.unquote(baidu_url)
# print(query_string)


from urllib import request,parse
# 1.拼url地址
url = 'https://www.baidu.com/s?wd={}'
word = input('请输入搜索内容:')
params = parse.quote(word)
full_url = url.format(params)
# 2.发请求保存到本地
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36'}
req = request.Request(url=full_url,headers=headers)
res = request.urlopen(req)
html = res.read().decode('utf-8')
# 3.保存文件至当前目录
filename = word + '.html'
with open(filename,'w',encoding='utf-8') as f:
    f.write(html)