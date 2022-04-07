import requests as req
import smtplib, ssl
resp = req.get("https://www.spacefox.shop/fr/")

print(resp.text)