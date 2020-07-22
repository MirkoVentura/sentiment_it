from bs4 import BeautifulSoup
import requests
import pandas as pd
from bs4 import BeautifulSoup
from html_parser import HTMLTableParser

if __name__ == "__main__":
	hp = HTMLTableParser()
	table = hp.parse_url("https://emojiterra.com/it/punti-di-codice/")[0][1] # Grabbing the table from the tuple
	print(table.head())
	table.to_csv('./emojitalia.csv')


