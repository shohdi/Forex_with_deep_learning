import smtplib
from email.mime.text import MIMEText
from security.email_login import EmailUser

subject = "Email Subject"
body = "This is the body of the text message"
sender = EmailUser["username"]
recipients = ["shohdi@gmail.com"]
password = EmailUser["password"]


def send_email(subject, body, sender, recipients, password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")


send_email(subject, body, sender, recipients, password)
