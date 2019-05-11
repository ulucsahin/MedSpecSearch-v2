from urllib import request
from bs4 import BeautifulSoup
import csv
import json

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

#file_name = "iCliniqDataNew.csv" # Collected data will be saved into this file.
def get_info(link):
    pageFile = request.Request(link, headers=hdr)
    pageFile = request.urlopen(pageFile)
    pageHtml = pageFile.read()
    pageFile.close()
    
    try:
        pageHtml = pageHtml.decode("utf-8")
    except:
        try:
            pageHtml = pageHtml.decode("ISO-8859-1")
        except:
            return {'title': " ", 'category1': " ", 'category2': " ",
                            'category3':" ", 'question':  " ", 'doctorName':  " ",
                            'doctorProfession':  " ", 'answers':  " "}
        
    soup = BeautifulSoup("".join(pageHtml))

    #sAll = soup.findAll("li")
    questionTitle = soup.find('h1', {'class': 'soft-margin'}).text[3:]

    question_soup = soup.find('div', {'class':'qContent'})
    question = question_soup.find_all('p')
    question = ' '.join([str(que) for que in question])


    cats_soup = soup.find('div', {'class': 'breadcrumb'})
    cats = cats_soup.find_all('span')
    category1 = cats[2].text[:-2]
    category2 = cats[3].text[:-2]
    category3 = cats[4].text[:-2]


    doctorInfo = soup.find('div', {'class': 'col-xs-9 col-sm-10 col-md-10 col-lg-10'})
    doctorName = doctorInfo.find_all('a')[0].text
    doctorProfession = doctorInfo.find('span', {'style': 'font-size:80% !important; color:#b71c1c;'}).text

    ansList = soup.find_all('div' , {'style': 'padding:15px 0 15px 0 !important;'})
    answers = []
    for ans in ansList:
        divCount = ans.find_all('div', {'class' : 'row'})
        raw = ans.find_all('p')
        if len(divCount)==0:
            answers.append(('Patient', raw))
        else:
            answers.append(('Doctor', raw[1:-1]))
         
    result = {'title': questionTitle, 'category1': category1, 'category2': category2,
                            'category3': category3, 'question': question, 'doctorName': doctorName,
                            'doctorProfession': doctorProfession, 'answers': answers}
    return result


def write_to_file(path, data, writer):
    f = open(path, "a", encoding="utf8")
    
    writer.writerows(data)
    f.close()


def start_collecting_data(file_name, starting_page):
    question_amount = -1
    page = starting_page # Script will start collecting data from this page number.
    f = open(file_name, "a", encoding="utf8")
    writer = csv.DictWriter(
        f, fieldnames=["title", "category1", "category2",
                       "category3", "question", 
                       "doctorName","doctorProfession", "answers"])

    writer.writeheader()

    while question_amount != 0: # safeguard
        page_data = []
        site = "https://www.icliniq.com/tools/tagLinks?page=" + str(page)
        req = request.Request(site, headers=hdr)
        
        content = request.urlopen(req)
        content = content.read()
        content = content.decode('utf-8')
        content = json.loads(content)
        
        question_amount = content["found"]
        
        if question_amount == 0:
            print("Finished all pages.")
            break

        for i in range(question_amount):
            link = content["items"][i]["url"]
            page_data.append(get_info(link))
        
        write_to_file(file_name, page_data, writer)
    #         with open("iCliniqData" + str(page) + ".json", 'w') as fout:
    #             json.dump(results, fout)
        
        page += 1
        print("Page {} finished. Continuing.".format(page-1), end = "\r")