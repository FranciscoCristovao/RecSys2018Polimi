# Notification on finishing
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class NotifyMail:
    def __init__(self, to_address, dataframe, text='A message from RS', subject='RS evaluation'):
        if dataframe is not None:
            self.data = dataframe
        else:
            # todo throw error
            print('Dataframe was null. Inserting a dummy df for test')
            l1 = [{'nada': 0, 'nisba': 1}, {'nada': 2, 'nisba': 3}]
            self.data = pd.DataFrame(l1, columns=['nada', 'nisba'])

        self.from_address = 'dummy.alessandro.artoni@gmail.com'
        self.pw = 'dummyalessandro1995'
        if to_address is 'arto':
            self.to_address = 'artonialessandro1995@gmail.com'
        elif to_address is 'francisco':
            self.to_address = 'idk'
        else:
            print('You should choose between -arto- or -francisco-')
            self.to_address = 'artonialessandro1995@gmail.com'
        self.text = text
        self.subject = subject

    def send_email(self):
        to = self.to_address
        html = self.data.to_html()

        # Gmail Sign In
        gmail_sender = self.from_address
        gmail_passwd = self.pw

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_sender, gmail_passwd)

        msg = MIMEMultipart('alternative')
        msg['Subject'] = self.subject
        msg['From'] = gmail_sender
        msg['To'] = to

        part1 = MIMEText(self.text, 'plain')
        part2 = MIMEText(html, 'html')

        msg.attach(part1)
        msg.attach(part2)

        try:
            server.sendmail(gmail_sender, to, msg.as_string())
            print('email sent')
        except:
            print('error sending mail')

        server.quit()
