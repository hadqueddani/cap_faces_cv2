
import cv2
import os
import imutils
import io
from ftplib import FTP, error_perm
from tqdm import tqdm

# FTP connection data
host = 'ftp.host.com'
port = 21
username = 'userg@user.com'
password = 'password'

# Data for FTP server
remote_directory_base = '/cap/data'
remote_directory_person = ''

personName = 'Daniel_Lozano'
dataPath = 'C:/Users/Pc/Documents/HandgunDataset/Reconocimiento Facial/data'
personPath = dataPath + '/' + personName

total_images = 100

# Connect to FTP
try:
    ftp = FTP()
    ftp.connect(host, port)
    ftp.login(username, password)

    # Check if the base directory exists
    remote_directory_person = f'{remote_directory_base}{personName}/'
    try:
        ftp.cwd(remote_directory_person)
    except error_perm as e_perm:
        if "550" in str(e_perm):
            # The directory does not exist, try to create it
            ftp.mkd(remote_directory_person)
            ftp.cwd(remote_directory_person)
        else:
            raise e_perm  # Retry

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    progress_bar_creation = tqdm(total=total_images, desc='Subiendo imágenes al FTP')

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Convert image to bytes
            _, img_encoded = cv2.imencode('.jpg', rostro)
            
            # Create a BytesIO object to read the binary data
            img_stream = io.BytesIO(img_encoded)

            # Upload the data to the person's folder
            ftp.storbinary(f'STOR {personName}_rostro_{count}.jpg', img_stream)

            count += 1
            progress_bar_creation.update(1)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= total_images:
            break

    progress_bar_creation.close()

    cap.release()
    cv2.destroyAllWindows()
    ftp.quit()

except Exception as e:
    print(f"Error al conectar o realizar operaciones FTP: {str(e)}")