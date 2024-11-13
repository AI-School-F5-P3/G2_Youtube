from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time


class YouTubeCommentScraper:
    def __init__(self):
        self.driver = webdriver.Chrome()

    def scroll_to_load_comments(self, scroll_times=10):
        last_height = self.driver.execute_script(
            "return document.documentElement.scrollHeight")

        for _ in range(scroll_times):
            self.driver.execute_script(
                "window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script(
                "return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def scrape_comments(self, video_url, num_comments=100):
        try:
            print(f"Scraping comments from {video_url}")
            self.driver.get(video_url)
            time.sleep(3)
            print("Scrolling to load comments...")
            self.scroll_to_load_comments()

            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "ytd-comment-thread-renderer"))
            )

            comments_data = []
            processed_comments = 0

            while len(comments_data) < num_comments:
                # obtain all comments on the page
                comment_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "ytd-comment-thread-renderer"
                )

                # procces new comments
                for comment in comment_elements[processed_comments:]:
                    try:
                        # extract information from the comment
                        comment_text = comment.find_element(
                            By.CSS_SELECTOR, "#content-text").text.strip()
                        comment_author = comment.find_element(
                            By.CSS_SELECTOR, "#author-text").text.strip()
                        comment_date = comment.find_element(
                            By.CSS_SELECTOR,
                            "#header-author .published-time-text").text.strip()

                        # append the comment data to the list
                        comments_data.append({
                            "comment": comment_text,
                            "author": comment_author,
                            "date": comment_date,
                        })

                        if len(comments_data) >= num_comments:
                            break
                    except NoSuchElementException as e:
                        print(f"Error processing comment. Skipping... {e}")
                        continue

                processed_comments = len(comment_elements)
                print(f"Processed {len(comments_data)} comments so far...")

                if len(comments_data) < num_comments:
                    self.driver.execute_script(
                        "window.scrollTo(0, document.documentElement.scrollHeight);")
                    time.sleep(2)

            return pd.DataFrame(comments_data)
        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()

    def close(self):
        self.driver.quit()


def main():
    video_url = "https://www.youtube.com/watch?v=_TqMek9evXs&ab_channel=NowThisImpact"

    scraper = YouTubeCommentScraper()
    try:
        # extract comments from the video
        print("Starting to scrape comments...")
        comments_df = scraper.scrape_comments(
            video_url=video_url,
            num_comments=100
        )
        if not comments_df.empty:
            # save the comments to a CSV file
            file_name = video_url.split("=")[-1] + "_comments.csv"
            comments_df.to_csv(file_name, index=False, encoding="utf-8")
            print(f"Comments saved to {file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

        print("\nFirst 10 comments:")
        print(comments_df.head(10))
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
