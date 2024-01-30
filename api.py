from flask import Flask, request, jsonify
from PIL import Image
from predict_plant_type import predict_type as plant
from predict_soil_type import predict_type as soil
import mysql.connector
import hashlib

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

soil_feature = {}
soil_feature['Black Soil'] = "Kara Toprak, genellikle Türkiye'nin Güneydoğu Anadolu Bölgesi'nde bulunan verimli bir toprak türüdür. Bu toprak, genellikle organik madde zenginliği, iyi drenaj özellikleri ve mineraller açısından zengin yapısıyla bilinir. Kara Toprak, bitkilerin büyümesi için gerekli besin maddelerini içerdiği için tarım için oldukça elverişlidir. Bu toprak türünde yetişebilen meyveler arasında üzüm, nar, incir, zeytin ve şeftali bulunmaktadır. Ayrıca sebzeler arasında ise domates, biber, patlıcan, salatalık gibi birçok farklı türde sebze yetişebilir. Kara Toprak, iklim koşulları ve toprak özellikleri nedeniyle tarım açısından önemli bir potansiyele sahiptir ve bu bölgede çeşitli meyve ve sebzelerin başarılı bir şekilde yetiştirilmesine olanak tanır."
soil_feature['Cinder Soil'] = "Cinder Soil veya Lav Taşı Toprağı, volkanik aktivitelerin sonucunda oluşan bir toprak türüdür. Genellikle hafif ve taneli bir yapıya sahip olan Cinder Soil, mineral zenginliği ile dikkat çeker. Bu toprak, içerdiği mineraller nedeniyle bitkiler için önemli bir besin kaynağı sağlar. Cinder Soil'de yetişebilen meyveler arasında çilek, ahududu, üzüm ve ananas gibi meyveler bulunabilir. Aynı zamanda marul, ıspanak, havuç ve patates gibi sebzeler de Cinder Soil'de başarılı bir şekilde yetiştirilebilir. Bu toprak türü, su tutma kapasitesi düşük olduğu için iyi bir drenaja sahiptir, bu da köklerin sağlıklı bir şekilde gelişmesine yardımcı olur. Cinder Soil, özellikle volkanik bölgelerde bulunur ve bu bölgelerde tarım açısından önemli bir rol oynar."
soil_feature['Laterite Soil'] = "Laterit toprak, genellikle tropikal ve subtropikal iklim bölgelerinde bulunan, koyu kırmızı veya turuncu renkte bir sertleşmiş tabakaya sahip olan bir toprak türüdür. Bu toprak, yüksek demir ve alüminyum oksit konsantrasyonlarına sahiptir ve su tutma kapasitesi düşük olmasına rağmen iyi bir drenaj özelliği gösterir. Laterit topraklar, bitkilerin büyümesi için gerekli olan besin maddelerini içerir, ancak organik madde bakımından zengin değildir. Bu toprak türünde yetişebilen meyveler arasında ananas, muz, mango ve guava gibi tropikal meyveler bulunurken, tatlı patates, hindistancevizi ve zencefil gibi sebzeler de başarıyla yetişebilir. Laterit topraklar, iklim koşulları ve mineral içeriği nedeniyle tarım açısından önemli bir potansiyele sahiptir, ancak düşük su tutma kapasitesi sulama ihtiyaçlarını dikkate almayı gerektirebilir."
soil_feature['Peat Soil'] = "Turba Toprağı, genellikle sulak ve bataklık alanlarda bulunan, yavaşça çürümüş bitki materyallerinin birikimi ile oluşan kahverengi veya siyah renkte bir organik toprak türüdür. Yüksek su tutma kapasitesine sahip olan Turba Toprağı, bahçe yetiştiriciliği ve tarım uygulamalarında sıkça kullanılır. Ancak, mineral içeriği düşük olduğu için besin maddeleri bakımından sınırlıdır. Yosunlar, odun çiçekleri, orkide ve asidik toprak seven bitkiler gibi bitkiler, bu toprak türünde başarıyla yetişebilir. Turba ekosistemlerinin korunması büyük önem taşır, çünkü turba oluşumu yıllar alabilir ve bataklık alanların ekosistem sağlığı için kritik bir rol oynar. Turba kullanımı, sürdürülebilir ve çevre dostu yöntemlerle gerçekleştirilmelidir, çünkü turba çıkartılması ekosistem bozulmasına ve karbon salınımına yol açabilir. Bu nedenle, turba toprağı kullanımı dikkatlice yönetilmelidir"
soil_feature['Yellow Soil'] = "Sarı Toprak veya Yellow Soil, genellikle ılıman ve yarı kurak iklim bölgelerinde bulunan bir toprak türüdür. Bu toprak, genellikle sarı renkte olan kumlu, killi veya tınlı bir yapıya sahiptir. Sarı Toprak, su tutma kapasitesi genellikle orta düzeyde olan ve iyi bir drenaj sağlayan bir karakteristik gösterir. Bitkiler için gerekli olan temel mineralleri içermesi nedeniyle tarım için uygun bir zemin olarak kabul edilir. Sarı Toprak'ta yetişebilen bitkiler arasında buğday, arpa, pamuk, soya fasulyesi gibi tarım ürünleri ve çeşitli meyve ağaçları bulunabilir. Bu toprak türü, iklim koşullarına ve coğrafi konumuna bağlı olarak farklı özelliklere sahip olabilir, ancak genel olarak tarım açısından kullanışlıdır ve bitki yetiştiriciliği için uygun bir ortam sağlar."

# MySQL bağlantı bilgileri
db = mysql.connector.connect(
    host='localhost',
    user='user',
    password='password',
    database='database'
)

# MySQL cursor oluştur
cursor = db.cursor()

@app.route('/soiltype', methods=['POST'])
def upload_image():
    try:
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        image = Image.open(image_file)
        image.save('uploads/' + image_file.filename)
        type_soil = soil()
        print(type_soil)
        return jsonify({'type': type_soil,
                        'feature' : soil_feature[type_soil]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/disease', methods=['POST'])
def disease_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        image = Image.open(image_file)
        image.save('uploads/' + image_file.filename)
        type_soil = plant()
        print(type_soil)
        return jsonify({'type': type_soil}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/signin', methods=['POST'])
def authenticate_user():
    try:
        data = request.get_json()

        username = data['username']
        password = data['password']
        print(username)
        print(password)

        hashed_login_password = hashlib.md5(password.encode()).hexdigest()

        query = 'SELECT * FROM users WHERE username = %s AND password = %s'
        cursor.execute(query, (username, hashed_login_password))

        result = cursor.fetchall()

        if len(result) > 0:
            return jsonify({'success': True,'message' : 'Logged in successfully'})
        else:
            return jsonify({'success': False, 'message' : 'Username or Password is incorrect'}), 401

    except Exception as e:
        print('Hata:', e)
        return jsonify({'error': 'Sunucu hatası'}), 500
    
@app.route('/signup', methods=['POST'])
def signup_user():
    try:
        data = request.get_json()

        username = data['username']
        password = data['password']

        # Kullanıcı adının daha önce oluşturulup oluşturulmadığını kontrol et
        check_query = 'SELECT * FROM users WHERE username = %s'
        cursor.execute(check_query, (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            return jsonify({'error': 'This username is already in use. Please choose another username.'}), 400

        # Şifreyi MD5 ile hashleme
        hashed_password = hashlib.md5(password.encode()).hexdigest()

        # MySQL sorgusu: Kullanıcıyı ekleyin
        insert_query = 'INSERT INTO users (username, password) VALUES (%s, %s)'
        cursor.execute(insert_query, (username, hashed_password))

        # Değişiklikleri kaydet
        db.commit()

        return jsonify({'success': True, 'message': 'User registration successful'})

    except Exception as e:
        print('Error:', e)
        return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
