import pandas as pd
import re
import os

def clean_reviews(input_file, output_file):
    """
    Görüşleri temizler:
    1. Başında ve sonunda tırnak işareti varsa kaldırır
    2. Birden fazla satırdan oluşan yorumları tek satıra indirir
    3. Her durum için tek satır formatında çıktı verir
    """
    
    def clean_single_review(text):
        """
        Tek bir görüşü temizler
        """
        if pd.isna(text) or text == "":
            return ""
        
        # String'e çevir
        text = str(text)
        
        # Tüm tırnak işaretlerini kaldır (başta, sonda ve ortada)
        text = text.replace('"', '')
        
        # Çok satırlı metni tek satıra çevir
        # Yeni satır karakterlerini boşlukla değiştir
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\r+', ' ', text)
        
        # Birden fazla boşluğu tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)
        
        # Başında ve sonundaki boşlukları kaldır
        text = text.strip()
        
        return text
    
    # Dosyayı oku
    try:
        df = pd.read_csv(input_file, encoding='utf-16')
    except:
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            df = pd.read_csv(input_file, encoding='latin-1')
    
    # Görüş sütununu temizle
    df['Görüş'] = df['Görüş'].apply(clean_single_review)
    
    # Boş görüşleri filtrele
    df = df[df['Görüş'].str.len() > 0]
    
    # Çıktı dosyasının dizinini oluştur
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Temizlenmiş veriyi kaydet
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Dosyayı tekrar oku ve tırnak işaretlerini kaldır
    with open(output_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # Tüm tırnak işaretlerini kaldır
    content = content.replace('"', '')
    
    # Dosyayı tekrar yaz
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write(content)
    
    print(f"Temizleme tamamlandı!")
    print(f"Temizlenmiş satır sayısı: {len(df)}")
    print(f"Çıktı dosyası: {output_file}")
    
    return df

def main():
    """
    Ana fonksiyon - örnek kullanım
    """
    # Script'in bulunduğu dizini bul ve dosya yollarını ayarla
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..', '..')
    
    # Proje kök dizinine geç
    os.chdir(project_root)
    
    input_file = "duyguanalizi/data/raw/sample2_data.csv"
    output_file = "duyguanalizi/data/processed/cleaned_reviews.csv"
    
    cleaned_df = clean_reviews(input_file, output_file)
    
    # İlk 5 temizlenmiş örneği göster
    print("\nİlk 5 temizlenmiş örnek:")
    print(cleaned_df.head())
    
    # Örnek temizleme sonuçları
    print("\nÖrnek temizleme sonuçları:")
    for i, (review, status) in enumerate(zip(cleaned_df['Görüş'][:3], cleaned_df['Durum'][:3])):
        print(f"\nÖrnek {i+1}:")
        print(f"Durum: {status}")
        print(f"Görüş: {review[:100]}...")

if __name__ == "__main__":
    main() 