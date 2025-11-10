# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_book_links(text):
    soup = bs4.BeautifulSoup(text, features='lxml')
    out = []
    for art in soup.select('article.product_pod'):
        rtag = art.find('p', class_='star-rating')
        classes = rtag.get('class', []) if rtag else []
        stars_name = next((c for c in classes if c in ['One', 'Two', 'Three', 'Four', 'Five']), None)
        star_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
        stars = star_map.get(stars_name, 0)
        ptxt = art.select_one('.price_color').get_text(strip=True) if art.select_one('.price_color') else ''
        digits = ''.join(ch for ch in ptxt if ch.isdigit() or ch == '.')
        price = float(digits) if digits else float('inf')
        if stars >= 4 and price < 50:
            href = (art.h3.a.get('href') or '').strip()
            parts = href.split('/')
            href_norm = '/'.join(parts[-2:]) if len(parts) >= 2 else href
            out.append(href_norm)
    return out


def get_product_info(text, categories):
    soup = bs4.BeautifulSoup(text, features='lxml')
    info = {}
    table = soup.find('table', class_='table')
    if table:
        for tr in table.find_all('tr'):
            th = tr.find('th').get_text(strip=True)
            td = tr.find('td').get_text(strip=True)
            info[th] = td
    title_tag = soup.select_one('div.product_main h1')
    title = title_tag.get_text(strip=True) if title_tag else ''
    rtag = soup.find('p', class_='star-rating')
    rclasses = rtag.get('class', []) if rtag else []
    rating_word = next((c for c in rclasses if c in ['One', 'Two', 'Three', 'Four', 'Five']), '')
    anchors = soup.select('ul.breadcrumb li a')
    category = anchors[-1].get_text(strip=True) if anchors else ''
    desc = ''
    hdr = soup.select_one('#product_description')
    if hdr:
        p = hdr.find_next_sibling('p')
        if p:
            desc = p.get_text(strip=True)
    if category not in categories:
        return None
    info.update({'Category': category, 'Rating': rating_word, 'Description': desc, 'Title': title})
    return info


def scrape_books(k, categories):
    rows = []
    list_base = 'http://books.toscrape.com/catalogue/page-{}.html'
    product_base = 'http://books.toscrape.com/catalogue/'
    for page in range(1, k + 1):
        resp = requests.get(list_base.format(page))
        if resp.status_code != 200:
            continue
        links = extract_book_links(resp.text)
        for rel in links:
            url = product_base + rel.lstrip('/')
            r = requests.get(url)
            if r.status_code != 200:
                continue
            row = get_product_info(r.text, categories)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)




# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def get_comments(storyid):
    base = 'https://hacker-news.firebaseio.com/v0/item/{}.json'
    def fetch(i):
        r = requests.get(base.format(i))
        return r.json() if r.status_code == 200 else None

    root = fetch(storyid)
    out = []

    def dfs(cid):
        item = fetch(cid)
        if not item or item.get('dead'):
            return
        if item.get('type') == 'comment':
            out.append({
                'id': item.get('id'),
                'by': item.get('by'),
                'text': item.get('text'),
                'parent': item.get('parent'),
                'time': pd.to_datetime(item.get('time', 0), unit='s')
            })
        for kid in item.get('kids', []):
            dfs(kid)

    for kid in root.get('kids', []):
        dfs(kid)

    return pd.DataFrame(out, columns=['id', 'by', 'text', 'parent', 'time'])
