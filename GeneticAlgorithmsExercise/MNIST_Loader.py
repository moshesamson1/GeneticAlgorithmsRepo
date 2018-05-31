from mnist import MNIST
from NetworkBackPropagation import *
from random import seed, sample
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from os.path import basename
import os
import smtplib


def get_train_images():
    mndata = MNIST('resources')
    mndata.gz = True
    images, labels = mndata.load_training()
    return images, labels


def create_csv_from_data(images, labels, filename, max_lines=-1):
    lines = []
    for i in xrange(len(labels)):
        to_add = list(images[i])
        to_add.extend([labels[i]])
        lines.append(','.join(str(x) for x in to_add))
    with open('resources/'+filename, 'wb') as csvfile:
        for line in (lines if max_lines == -1 else sample(lines, max_lines)):
            csvfile.write(line + '\n')


def send_results_via_email(text, title):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("moshe.samson@mail.huji.ac.il", "moshe_samson770")

    msg = MIMEMultipart()
    msg['From'] = "moshe.samson@mail.huji.ac.il"
    msg['To'] = COMMASPACE.join("samson.moshe@gmail.com")
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = title

    msg.attach(MIMEText(text))

    # for f in [title+"_max_path.png", title+"_fcc_hmValues_graph.png"]:
    #     with open(f, "rb") as fil:
    #         part = MIMEApplication(
    #             fil.read(),
    #             Name=basename(f)
    #         )
    #     # After the file is closed
    #     part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
    #     msg.attach(part)
    server.sendmail("moshe.samson@mail.huji.ac.il", "samson.moshe@gmail.com", msg.as_string())
    server.quit()