import smtplib
# import imghdr
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

Sender_Email = "traffic80085@gmail.com"
Password = "Helloworld@5"


Reciever_Email ="traffic80085@gmail.com"# "hocav89528@v2ssr.com"


def sendmail(image_path,numplate_list):


    fromaddr = "traffic80085@gmail.com"  # Your Gmail address
    toaddr = "traffic80085@gmail.com"  # Recipient's email address

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "!!! VOILATION !!! NO HELMET FOUND FOR INDEX"

    body = "Find the attached image of violation and approved." 
    body += "\nDetected Number Plates: " + ", ".join(numplate_list)


    msg.attach(MIMEText(body, 'plain'))

    filename = "filename.jpg"
    attachment = open(image_path, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "tkyd pvhc qbbu yxzl")  # Enter your Gmail password or app password here
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
def sendmail1(image_path):


    fromaddr = "traffic80085@gmail.com"  # Your Gmail address
    toaddr = "traffic80085@gmail.com"  # Recipient's email address

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "!!! VOILATION !!! NO HELMET FOUND FOR INDEX"

    body ="!!! VOILATION !!! NO HELMET FOUND FOR INDEX"

    msg.attach(MIMEText(body, 'plain'))

    filename = "filename.jpg"
    attachment = open(image_path, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "tkyd pvhc qbbu yxzl")  # Enter your Gmail password or app password here
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()