import requests
from bs4 import BeautifulSoup
import csv


# # Collect and parse first page
# page = requests.get('https://www.goldpriceindia.com/gold-price-' + month + '-2018.php')
# soup = BeautifulSoup(page.text, 'html.parser')

# # Pull all text from the BodyText div
# # div_class = soup.find(class_='panel-body')

# # div_class_gold_title = soup.find_all(class_ = 'gold-panel-title')

# h4_title = soup.find_all('h4')

# list_h4 =[]
# for i in h4_title :
#     value = i.text
#     list_h4.append(value)

# print(list_h4)

# table_data = soup.find_all(class_= 'table-data dayyeartable')
# # table_head = table_data.find_all('thead')

# # th_tags = table_head.find_all('th')

# # table_body = table_data.find_all('tbody')

# # tr_body = table_body.find_all('tr')
# # print(table_data)

# with open('dataset.csv','w') as csvfile:
#     csvwriter = csv.writer(csvfile , delimiter =',', quoting=csv.QUOTE_MINIMAL )
#     count =0
#     for td in table_data:

#         th = td.find_all('th')
#         # add that to csv
#         l =[]
#         tds = td.find_all('td')
#         m =[]
#         for i in tds:
#             value = i.text
#             m.append(value.encode('ascii', 'ignore').decode('ascii'))
#         if(len(m)==4):
#             continue
#         for i in th:
#             l.append(i.text)
        
#         csvwriter.writerow([str(list_h4[count])[14:], m[1]])
#         # csvwriter.writerow(l)

#         print(m)
#         # csvwriter.writerow(m[:4])
#         # csvwriter.writerow(m[-4:])
#         count = count +1
    

month_list =['january','february','march','april','may','june','july','august','september','october','november','december']
year ='2011'
for month in month_list:

    page = requests.get('https://www.goldpriceindia.com/gold-price-' + month + '-'+year+'.php')
    soup = BeautifulSoup(page.text, 'html.parser')


    h4_title = soup.find_all('h4')

    list_h4 =[]
    for i in h4_title :
        value = i.text
        list_h4.append(value)

    print(list_h4)

    table_data = soup.find_all(class_= 'table-data dayyeartable')


    with open('dataset'+year+'.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile , delimiter =',', quoting=csv.QUOTE_MINIMAL )
        count =0
        for td in table_data:

            th = td.find_all('th')
        # add that to csv
            l =[]
            tds = td.find_all('td')
            m =[]
            for i in tds:
                value = i.text
                m.append(value.encode('ascii', 'ignore').decode('ascii'))
            if(len(m)==4):
                continue
            for i in th:
                l.append(i.text)

            date = str(list_h4[count])[14:]
            day = date[:2]
            year = date[-4:]
            dicti = {'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06','july':'07','august':'08','september':'09'
            ,'october':'10','november':'11','december':'12'}
            for key,value in dicti.items():
                if(key == month):
                    month_value = value
            p=int(m[1].replace(",",""))
            csvwriter.writerow([day+'-'+month_value+'-'+year, p])
        # csvwriter.writerow(l)

            print(m)
        # csvwriter.writerow(m[:4])
        # csvwriter.writerow(m[-4:])
            count = count +1
    



