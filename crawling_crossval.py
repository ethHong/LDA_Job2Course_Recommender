import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib
from urllib import request
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os
from fake_useragent import UserAgent
from tqdm.notebook import trange
import time

class JDcrawler_recommender_crossval():

    def __init__(self, name, driverpath, driver, options, topicnum, keyword):
        print("크롤러 초기 설정중...")
        #self.ID = input("ID:")
        #self.PASS = input("PASS: ")
        self.driverpath = driverpath
        self.driver = driver
        self.options = options
        self.name = name
        self.topicnum = topicnum
        self.keyword = keyword
        self.keyword = ",".join(self.keyword.split()).lower()

    def load_processed(self):
        processed = pd.read_csv(
            os.getcwd() + "/train_dataset" + "/processed_courses_data_{}topic.csv".format(self.topicnum))
        return processed

    def mock_user_agent(self):
        ua = UserAgent()
        working = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15"
        working_tail = "(" + working.split("(")[-1]
        random_head = ua.random.split("(")[0] + "(" + ua.random.split("(")[1]
        return random_head + working_tail

    def check_http_error(self):
        return "HTTP ERROR 429" in self.driver.page_source

    def login_linkedin(self):
        self.ID = input("ID:")
        self.PASS = input("PASS: ")
        self.options.add_experimental_option('excludeSwitches', ['enable-automation'])

        userAgent = self.mock_user_agent()
        self.options.add_argument(f'user-agent={userAgent}')
        self.options.add_argument('headless')
        driverpath = os.getcwd() + "/chromedriver"
        wait = WebDriverWait(self.driver, 10)

        url = "https://www.linkedin.com/"
        self.driver.get(url)

        # driver.find_element_by_xpath('/html/body/div/main/p/a').click()
        try:
            elem = self.driver.find_element_by_xpath('//*[@id="session_key"]')
            elem.send_keys(self.ID)
            elem = self.driver.find_element_by_xpath('//*[@id="session_password"]')
            elem.send_keys(self.PASS)

            self.driver.find_element_by_xpath('/html/body/main/section[1]/div[2]/form/button').click()
        except:
            if self.check_http_error() == True:
                print("Take a break...for 3 minuits...")
                time.sleep(180)

                elem = self.driver.find_element_by_xpath('//*[@id="session_key"]')
                elem.send_keys(self.ID)
                elem = self.driver.find_element_by_xpath('//*[@id="session_password"]')
                elem.send_keys(self.PASS)

                self.driver.find_element_by_xpath('/html/body/main/section[1]/div[2]/form/button').click()

    def refresh_link(self, continue_link):
        userAgent = self.mock_user_agent()
        self.options.add_argument(f'user-agent={userAgent}')
        self.options.add_argument('headless')
        self.driverpath = os.getcwd() + "/chromedriver"
        self.driver = webdriver.Chrome(self.driverpath, chrome_options=self.options)

        self.login_linkedin()
        self.driver.get(continue_link)

    def refine(self, c):
        c_ref = "-".join(c.split(" ")).lower()
        return c_ref

    def refresh_source_pages(self):
        time.sleep(3)
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        try:
            p = soup.find("ul", {"class": "artdeco-pagination__pages artdeco-pagination__pages--number"}).find_all("li")
        except:
            self.driver.get(self.driver.current_url)
            time.sleep(5)
            p = soup.find("ul", {"class": "artdeco-pagination__pages artdeco-pagination__pages--number"}).find_all("li")
        return [p, soup]

    def crawl_jd(self):
        time.sleep(3)
        soup = self.refresh_source_pages()[1]


        potision = []
        job_details = []

        jobs = soup.find_all("li", {"class": "jobs-search-results__list-item occludable-update p0 relative ember-view"})
        jobs_id = [i["id"] for i in jobs]

        for i in jobs_id:
            self.driver.find_element_by_xpath('//*[@id="{}"]'.format(i)).click()

            self.driver.implicitly_wait(10)
            # refresh page source
            soup = self.refresh_source_pages()[1]

            self.driver.implicitly_wait(10)

            Position = soup.find("h2",
                                 {"class": "jobs-details-top-card__job-title t-20 t-black t-normal"}).text.rstrip()

            Job_Details = soup.find("div", {"id": "job-details"}).text.strip()
            potision.append(Position)
            job_details.append(Job_Details)

        return pd.DataFrame({"Position": potision, "Job_Details": job_details})

    def crawl_job_description(self, starting_page, how_many, total_page, start_url):
        if how_many > total_page:
            how_many = total_page

        self.driver.get(start_url)
        self.driver.implicitly_wait(10)
        time.sleep(3)
        pages = self.refresh_source_pages()[0]
        soup = self.refresh_source_pages()[1]

        current = starting_page
        df = pd.DataFrame()

        for i in trange(starting_page - 1, how_many + starting_page - 1):

            print("Crawling {} out of {} pages...".format(current, total_page))

            pages_meta = [j.text.strip().split()[0] for j in pages]

            # Do Crawling#

            crawed_page = self.crawl_jd()
            df = pd.concat([df, crawed_page])
            current = current + 1

            if current > total_page:
                break  # Don't move page if it's last page
            # Move page

            try:
                index_of_next_page = pages_meta.index(str(i + 2))
            except ValueError:
                index_of_next_page = len(pages_meta) - 1 - pages_meta[::-1].index('…')

            button_aria_label = pages[index_of_next_page].find("button")["aria-label"]

            # 해당 버튼이 나올때까지 기다려주기

            self.driver.implicitly_wait(10)

            try:
                self.driver.find_element_by_xpath('//*[@aria-label="{}"]'.format(button_aria_label)).click()

            except:
                self.driver.get(self.driver.current_url)
                self.driver.implicitly_wait(10)
                button_aria_label = str(int(button_aria_label.split()[0]) + 1) + " " + button_aria_label.split()[1]
                self.driver.find_element_by_xpath('//*[@aria-label="{}"]'.format(button_aria_label)).click()

            self.driver.implicitly_wait(10)
            print("Upcoming page is {}".format(i + 2))
            upcoming = self.driver.current_url
            # Refresh List
            try:
                pages = self.refresh_source_pages()[0]
            except:
                self.driver.get(self.driver.current_url)
                time.sleep(3)
                pages = self.refresh_source_pages()[0]

        return (df, upcoming)

    def crawl(self, keyword, counts, how_many=3):
        header = "https://www.linkedin.com/jobs/search/?geoId=105149562&keywords="
        link = header + self.refine(keyword)
        self.driver.get(link)

        time.sleep(3)
        # 밑에 self함수 불러와서 크롤
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        self.driver.implicitly_wait(10)

        pages = soup.find("ul", {"class": "artdeco-pagination__pages artdeco-pagination__pages--number"}).find_all("li")

        total_page = int(pages[-1].text.strip())
        start_url = self.driver.current_url

        df = pd.DataFrame()
        count = df.shape[0]

        for i in trange(total_page // how_many):
            while count < counts - 1:
                starting_page_num = 1 + (3 * i)

                try:
                    out = self.crawl_job_description(starting_page_num, how_many, total_page, start_url)

                except:
                    if self.check_http_error() == True:

                        if starting_page_num != 1:
                            start_url = out[1]

                        print("Take a break...for 3 minuits...")
                        time.sleep(180)
                        self.refrech_link(start_url)
                        out = self.crawl_job_description(starting_page_num, how_many, total_page, start_url)
                    else:
                        print("refresh due to error...")
                        self.refresh_link(start_url)
                        self.driver.implicitly_wait(10)
                        out = self.crawl_job_description(starting_page_num, how_many, total_page, start_url)
                start_url = out[1]
                df = pd.concat([df, out[0]])
                count = df.shape[0]
                print("Count: {}".format(count))
                print("Refreshing for {} times".format(i + 1))

                if count != counts - 1:
                    self.refresh_link(start_url)

        return df