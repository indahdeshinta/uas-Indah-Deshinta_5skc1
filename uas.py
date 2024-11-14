import cv2
import os

# Membuat folder untuk menyimpan gambar jika belum ada
output_folder = "foto_yang_disimpan"  # Mengubah nama folder menjadi foto_yang_disimpan
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Memuat haarcascades untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Menangkap video dari webcam (gunakan 0 untuk webcam default)
cap = cv2.VideoCapture(0)

# Pastikan webcam terbuka dengan benar
if not cap.isOpened():
    print("Error: Webcam tidak ditemukan.")
    exit()

# Ambil gambar satu kali
ret, frame = cap.read()

# Jika gambar berhasil diambil
if ret:
    # Deteksi wajah dalam gambar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Menandai wajah yang terdeteksi (untuk pengujian)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Menyimpan gambar original
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), frame)

        # Crop wajah dari gambar original
        face_cropped = frame[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_folder, "cropped_face.jpg"), face_cropped)

        # Convert ke grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, "grayscale.jpg"), gray_image)

        # Convert ke black and white (thresholding)
        _, bw_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_folder, "blackwhite.jpg"), bw_image)

        # Menampilkan gambar yang telah diambil
        cv2.imshow("Captured Image", frame)
        cv2.imshow("Cropped Face", face_cropped)

# Tunggu beberapa detik agar gambar bisa dilihat
cv2.waitKey(2000)

# Menutup webcam dan jendela
cap.release()
cv2.destroyAllWindows()

print("Gambar telah disimpan di folder:", output_folder)
