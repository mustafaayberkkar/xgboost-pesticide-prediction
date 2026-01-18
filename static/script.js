$(document).ready(function() {
    $('.searchable-select').select2();

    // Ürünleri dinamik filtrele
    $('#Pestisit').on('change', function() {
        const secilenPestisit = $(this).val();
        const urunSelect = $('#Urun');

        urunSelect.empty();
        urunSelect.append('<option value="" disabled selected>Seçiniz...</option>');

        if (productMap && productMap[secilenPestisit]) {
            productMap[secilenPestisit].forEach(function(urun) {
                urunSelect.append(new Option(urun, urun));
            });
            urunSelect.prop('disabled', false);
        } else {
            urunSelect.append('<option value="" disabled>Ürün bulunamadı</option>');
            urunSelect.prop('disabled', true);
        }
        urunSelect.trigger('change');
    });
});

// FORM GÖNDERİMİ
document.getElementById('aiForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const btn = document.getElementById('btnSubmit');
    const resultBox = document.getElementById('resultBox');
    const resTitle = document.getElementById('resTitle');
    const resText = document.getElementById('resText');
    const probBar = document.getElementById('probBar');
    
    // Bilgi alanları
    const mrlInfo = document.getElementById('mrlInfo');
    const vpInfo = document.getElementById('vpInfo');
    const kalintiInfo = document.getElementById('kalintiInfo');
    const indexInfo = document.getElementById('indexInfo');
    const tutunmaInfo = document.getElementById('tutunmaInfo');

    btn.disabled = true;
    btn.innerHTML = "⏳ Veriler İşleniyor...";
    resultBox.style.display = 'none';

    // Verileri Topla (Buhar Basıncı YOK - Arkaplanda)
    const formData = {
        'Pestisit': $('#Pestisit').val(),
        'Urun': $('#Urun').val(),
        'Sicaklik_C': document.getElementById('Sicaklik_C').value,
        'Uygulanan_Ilac_Miktari_ml': document.getElementById('Uygulanan_Ilac_Miktari_ml').value
    };

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        if(data.error) {
            alert(data.error);
            return;
        }

        resultBox.style.display = 'block';
        resText.innerText = data.result_text;
        
        // Verileri Göster
        mrlInfo.innerText = data.found_mrl;
        vpInfo.innerText = data.auto_vp; // Otomatik bulunan basınç
        kalintiInfo.innerText = data.calc_kalinti;
        indexInfo.innerText = data.calc_index;
        tutunmaInfo.innerText = data.auto_tutunma;

        probBar.style.width = data.confidence + '%';
        probBar.innerText = '%' + data.confidence + ' Güven';

        if(data.prediction === 1) {
            resTitle.innerText = "⚠️ DİKKAT: RİSKLİ";
            resTitle.style.color = "#c0392b";
            probBar.style.backgroundColor = "#c0392b";
        } else {
            resTitle.innerText = "✅ SONUÇ: GÜVENLİ";
            resTitle.style.color = "#27ae60";
            probBar.style.backgroundColor = "#27ae60";
        }
    })
    .catch(err => {
        console.error(err);
        alert("Bağlantı hatası oluştu!");
    })
    .finally(() => {
        btn.disabled = false;
        btn.innerHTML = "🛡️ Analizi Başlat";
    });
});