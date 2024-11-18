import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import argparse

WCONCEPT_URL_TMPL = "https://www.wconcept.com/search/result.html?sort=popular&query=%s&sale=&cfo=categories&offset=0&limit=80"

# Returns the url of the clothes by given query in WConcept shopping website.
def get_wconcept_clothes_url(query="white cardigan"):
	sanitized_query = query.replace(" ", "%20")
	return WCONCEPT_URL_TMPL % (query)

def click_next_page_if_available(driver):
	try:
		init_source = driver.page_source
		WebDriverWait(driver, 10).until(
	    	EC.element_to_be_clickable((By.XPATH, '//button[@class="next"]')))
		next_page_button=driver.find_element(By.XPATH, '//button[@class="next"]')
		next_page_button.click()
		time.sleep(5)
		print ('Has page changed?', driver.page_source != init_source)
		return True
	except:
		print('Cannot find next page button!')
		return False


def get_image_urls(soup):
	imgs = set()
	for img in soup.find_all('img'):
		img_url = img.get('data-src')
		if img_url and img_url.endswith("_1.png"):
			imgs.add(img_url)
			print(img_url)
	print('Found %d images from page! ' % len(imgs), '\n\n')
	return imgs


def ScrapeImages(query="white cardigan", folder="./images/cardigan", limit=200):
	# Get Url
	website_url = get_wconcept_clothes_url(query)
	print("Scraping images from website: ", website_url, '\n\n')

	# Scrape
	service = Service()
	options = webdriver.ChromeOptions()
	options.add_argument("--headless=new")
	driver = webdriver.Chrome(service=service, options=options)

	# Fetch images from initial page
	driver.get(website_url)
	WebDriverWait(driver, 40).until(
		EC.presence_of_element_located((By.ID, "yesplz-pagination"))
	)
	soup = BeautifulSoup(driver.page_source, 'html.parser')
	image_urls = get_image_urls(soup)

	next_page_found = click_next_page_if_available(driver)
	while (len(image_urls) < limit and next_page_found):
		soup = BeautifulSoup(driver.page_source, 'html.parser')
		image_urls.update(get_image_urls(soup))
		next_page_found = click_next_page_if_available(driver)

	# print(soup.prettify())  #for debugging
	driver.quit()

	# Save images
	print('Start to save the images!')
	count = 1
	for img_url in image_urls:
		resp = requests.get(img_url)
		filename = img_url.rpartition('/')[-1]
		f = open(os.path.join(folder,filename),'wb')
		f.write(resp.content)
		f.close()
		count += 1
	print('Success!! %d images found and saved: ' % count)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="white cardigan")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--folder", type=str, default="./images/cardigan")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    ScrapeImages(query=args.query, limit=args.limit, folder=args.folder)
