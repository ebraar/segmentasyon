import cv2
import numpy as np

# Görüntüyü oku
image = cv2.imread('130026.jpg')
cv2.imshow('orijinal resim', image)

# Görüntüyü yumuşat
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('yumusatilmis resim', blurred)

# HSV'ye dönüştür
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)

# Timsah rengi için HSV aralıkları
lower1 = np.array([20, 30, 30])
upper1 = np.array([35, 255, 255])
lower2 = np.array([0, 25, 25])
upper2 = np.array([15, 200, 200])

# Maskeleme işlemi
mask1 = cv2.inRange(hsv, lower1, upper1)
mask2 = cv2.inRange(hsv, lower2, upper2)
mask = cv2.bitwise_or(mask1, mask2)
cv2.imshow('mask resim', mask)

# Gürültüyü temizle
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow('gürültü', mask)

# Kenar tespiti
edges = cv2.Canny(blurred, 50, 150)
edges = cv2.dilate(edges, kernel, iterations=1)
cv2.imshow('kenar resim', edges)

# Kenarları maske ile birleştir
mask = cv2.bitwise_and(mask, mask, mask=edges)
cv2.imshow('kenarlar maske ile birles', mask)

# Kontürleri bul
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # Kontürleri filtrele
    image_center = (image.shape[1] // 2, image.shape[0] // 2)
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Şekil analizi
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Moment analizi
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Merkeze uzaklık
                distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
                
                # Filtreleme kriterleri:
                # 1. Minimum alan
                # 2. Dairesellik oranı (timsah uzun ve ince)
                # 3. Merkeze uzaklık
                # 4. Dikey pozisyon
                if (area > 2000 and 
                    circularity < 0.5 and  # Uzun ve ince şekiller
                    distance < image.shape[1] // 3 and 
                    cy > image.shape[0] // 3 and 
                    cy < image.shape[0] * 0.7):  # Üst ve alt sınır
                    
                    filtered_contours.append((contour, area))
    
    if filtered_contours:
        # En büyük uygun kontürü seç
        largest_contour = max(filtered_contours, key=lambda x: x[1])[0]
        
        # Yeni maske oluştur
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
        
        # Maskeyi hafifçe genişlet
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        cv2.imshow('final mask', final_mask)
        
        # Sadece timsahı göster
        result = cv2.bitwise_and(image, image, mask=final_mask)
        
        # Sonucu göster
        cv2.imshow('Timsah Segmentasyonu', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Sonucu kaydet
        cv2.imwrite('segmented_crocodile.jpg', result)