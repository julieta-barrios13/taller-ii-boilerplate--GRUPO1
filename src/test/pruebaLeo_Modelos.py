import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1️⃣ Cargar dataset balanceado
df = pd.read_csv("entregas semanales/comentarios_balanceados.csv")

# 2️⃣ Variables
X = df["Review_clean"]
y = df["Sentiment"]

# 3️⃣ Lista de stopwords personalizada (de tu compañero)
stopwords = [
'able','accept','accepted','access','accommodation','accommodations','accomodation',
'according','account','across','action','actual','actually','add','added','additional',
'address','advance','advertised','advertising','advice','advise','advised','agent',
'agents','ago','agree','agreed','ahead','air','airbb','airbnb','airbnbs','allow',
'allowed','allowing','almost','alone','along','already','also','alternative',
'although','always','amount','another','answer','answers','anyone','anything',
'anyway','anywhere','apartment','apartments','apparently','april','area','around',
'arrival','arrive','arrived','arriving','ask','asked','asking','assistance','attempt',
'august','automated','available','aware','away','back','bank','based','basic',
'basically','bathroom','bb','became','become','behind','believe','beyond','big','bit',
'block','building','business','calendar','call','called','calling','calls','came',
'car','card','care','careful','case','cases','cash','center','centre','chance',
'change','changed','charge','charged','charges','charging','chat','check','checked',
'checkin','checking','children','choice','choose','chose','city','claim','claimed',
'claiming','claims','clean','cleaned','cleaning','clear','clearly','clients','close',
'closed','code','come','comes','comfortable','coming','communicate','communication',
'community','companies','company','complete','completely','condition','conditions',
'confirm','confirmation','confirmed','consider','considering','consumer','contact',
'contacted','contacting','continue','contract','control','conversation','correct',
'cost','costs','could','country','couple','course','cover','covered','covid','credit',
'customer','customers','cut','date','dates','day','days','dealing','dealt','decided',
'decision','declined','delete','deleted','department','deposit','described',
'description','despite','details','difference','different','direct','directly',
'documentation','dollars','done','door','double','drive','due','early','easily',
'easy','either','else','elsewhere','email','emailed','emails','emergency','end',
'ended','english','enough','entire','especially','etc','euros','even','evening',
'eventually','ever','every','everyone','everything','everywhere','evidence',
'exactly','except','expect','expected','experience','experienced','experiences',
'explain','explained','explanation','extra','face','fact','family','far','fee','fees',
'find','finding','fine','first','five','fix','flat','flight','flights','floor',
'follow','followed','following','food','forward','found','four','free','friend',
'friends','front','full','fully','funds','furniture','future','get','gets','getting',
'give','given','giving','go','goes','going','gone','got','government','group','guest',
'guests','happen','happened','happens','hard','hear','heard','help','helped','helping',
'hold','holiday','home','homes','hope','host','hosted','hosting','hosts','hour','hours',
'house','however','hrs','human','husband','idea','immediately','important','included',
'including','info','information','informed','inside','instead','insurance','interest',
'interested','internet','involved','issue','issues','items','job','july','june','keep',
'keeping','kept','key','keys','kids','kind','kitchen','knew','know','lack','lady',
'landlord','large','last','late','later','least','leave','leaving','left','legal',
'less','let','level','life','like','line','link','list','listed','listing','listings',
'literally','little','live','living','local','location','lock','locked','london',
'long','longer','look','looked','looking','looks','lose','lost','lot','lots','low',
'made','main','major','make','makes','making','man','managed','management','manager',
'many','march','matter','may','maybe','mean','means','meant','meet','member','mention',
'mentioned','mess','message','messaged','messages','messaging','met','middle','might',
'mind','minute','minutes','money','month','months','morning','move','moved','much',
'multiple','must','name','near','nearly','need','needed','needs','new','next','night',
'nights','nobody','non','none','note','nothing','notice','noticed','notified','nowhere',
'number','numerous','obviously','offer','offered','offering','often','ok','old','one',
'ones','online','open','opened','option','options','order','original','others','outside',
'overall','owner','owners','page','paid','pandemic','paris','parking','part','partial',
'party','passport','past','pay','paying','payment','payments','paypal','people','per',
'period','person','personal','phone','photo','photos','picture','pictures','place',
'places','planned','plans','platform','please','plus','pm','pocket','point','police',
'policies','policy','pool','post','posted','previous','price','prices','prior','private',
'probably','process','professional','profile','proper','properly','properties','property',
'prove','provide','provided','providing','public','put','question','questions','quickly',
'quite','rate','rather','rating','reach','reached','read','reading','ready','reason',
'reasonable','reasons','receive','received','receiving','recent','recently','regarding',
'remove','removed','rent','rental','rentals','rented','renter','renters','renting',
'replied','reply','report','reported','representative','request','requested','requests',
'required','reservation','reservations','reserved','resolution','respect','respond',
'responded','response','responses','responsibility','responsible','result','return',
'returned','review','reviews','room','rooms','rules','run','running','safe','safety',
'said','save','see','seem','seemed','seems','seen','send','sending','sense','sent',
'service','services','set','several','share','shared','short','show','showed','shower',
'showing','shows','side','similar','simple','simply','since','single','site','sites',
'situation','sleep','small','someone','something','somewhere','soon','sort','space',
'speak','spend','spent','spoke','staff','standard','standards','star','stars','start',
'started','state','stated','states','stating','stay','stayed','staying','stays','still',
'stop','story','straight','street','submitted','superhost','support','supposed','sure',
'system','take','taken','takes','taking','talk','talking','team','tell','telling',
'terms','text','th','therefore','thing','things','think','thinking','third','though',
'thought','thousands','three','ticket','time','times','today','together','toilet',
'told','took','top','total','touch','towards','towels','town','trash','travel',
'traveling','travelling','treat','tried','trip','trouble','true','trust','try','trying',
'turn','turned','tv','twice','two','type','uk','unable','unless','unit','update','upon',
'use','used','user','users','using','vacation','value','verification','verified','verify',
'via','video','view','villa','visit','voucher','wait','waited','waiting','want','wanted',
'wants','warning','water','way','website','week','weekend','weeks','well','went',
'whatsoever','whether','whole','wife','wifi','willing','window','windows','within',
'without','word','work','worked','working','works','world','write','writing','written',
'year','years','yes','yet','young','zero','bnb','app','hostel','hostels','aircnc','air',
'book','booked','booking','bookingcom','bookings'
]

# 4️⃣ Vectorización TF-IDF con stopwords
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words=stopwords
)
X_vect = vectorizer.fit_transform(X)

# 5️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42, stratify=y
)

# 6️⃣ Entrenamiento con Regresión Logística
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# 7️⃣ Evaluación
y_pred = model.predict(X_test)
print("\n📊 Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\n", classification_report(y_test, y_pred))

# 8️⃣ Guardar modelo y vectorizador
joblib.dump(vectorizer, "src/test/vectorizador_tfidf.pkl")
joblib.dump(model, "src/test/modelo_lr.pkl")
print("\n✅ Modelo y vectorizador guardados en src/test/")
