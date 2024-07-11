from bs4 import BeautifulSoup
import requests
import csv
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

main_url = 'http://loveread.ec/'
base_url = 'http://loveread.ec/index_book.php?id_genre=1&p=85'
def increment_page(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'p' in query_params:
        current_page = int(query_params['p'][0])
        next_page = current_page + 1
        query_params['p'] = str(next_page)
        new_query_string = urlencode(query_params, doseq=True)
        new_url = urlunparse(parsed_url._replace(query=new_query_string))
        return new_url
    else:
        return url  # если параметр p не найден, возвращаем исходный URL
    
def parcing(num_books, output_csv='books.csv'):
    count = 0
    current_url = base_url
    headers = {
        "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
    }
    # Открываем CSV-файл для записи
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['page_url', 'image_url', 'author', 'title', 'annotation'])
        while count < num_books:
        #    print(f'Fetching URL: {current_url}')
            response = requests.get(current_url, headers=headers)
            soup = BeautifulSoup(response.text, 'lxml')
            # Найдем все блоки с книгами
            book_blocks = soup.find_all('tr', class_='td_center_color')
            for i in range(0, len(book_blocks), 2):
                if count >= num_books:
                    break
                book_info_block = book_blocks[i]
                book_annotation_block = book_blocks[i + 1]
                title_tag = book_info_block.find('a', title=True)
              #  if not title_tag:
              #      continue
                title = title_tag['title']
               # print(title)
                author_tag = book_info_block.find('a', href=lambda x: x and 'biography-author' in x)
              #  if not author_tag:
              #      continue
                author = author_tag.text.strip()
               # print(author)
                annotation = book_annotation_block.find('p').text.strip()
               # print(annotation)
                image_tag = book_info_block.find('img', class_='margin-right_8')
              #  if not image_tag:
               #     continue
                image_url = main_url + image_tag['src']
               # print(image_url)
               # if not book_url_tag:
                #    continue
                book_url_tag = book_info_block.find('a', href=lambda x: x and 'view_global.php?' in x)['href']
             #   print(book_url_tag)
                page_url = main_url + book_url_tag
             #   print(page_url)
                # Записываем данные в CSV-файл
                writer.writerow([page_url, image_url, author, title, annotation])
                count += 1
                # Каждые 10 книг делаем паузу на 10 секунд
                if count % 10 == 0:
                    time.sleep(10)
                    print(f'Парсинг в процессе, спарсил {count} книг')
            # Получаем URL следующей страницы
            current_url = increment_page(current_url)
          #  print(f'Next URL: {current_url}')
        print('Парсинг окончен')
        
parcing(1000, 'books_1000.csv')
