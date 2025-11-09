import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# 1) Cargar
df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")

# Mostrar columnas
print("Columnas del CSV:", list(df.columns))

# Elegir columna de texto disponible
text_col = "Review_clean" if "Review_clean" in df.columns else "Comment"

#stopwords = [...]  # Definir si es necesario
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

# Al no existir 'Sentiment', lo creamos desde 'Score'
if "Sentiment" in df.columns:
    y = df["Sentiment"].astype(str)
else:
    if "Score" not in df.columns:
        raise ValueError("No existen columnas 'Sentiment' ni 'Score' en el CSV.")
    # Asegurar que Score sea numérico
    score = pd.to_numeric(df["Score"], errors="coerce")

    # Filtrar filas válidas (score y texto presentes)
    mask = score.notna() & df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    df = df.loc[mask].copy()
    score = score.loc[mask]

    # Mapear estrellas → sentimiento
    def map_sentiment(s):
        if s <= 2: return "neg"
        elif s == 3: return "neu"
        else: return "pos"
    df["Sentiment"] = score.apply(map_sentiment)
    y = df["Sentiment"].astype(str)

# Texto (relleno por seguridad)
X_text = df[text_col].astype(str).fillna("")

# 2) Split (estratificado si hay suficientes ejemplos por clase)
from collections import Counter as Cn
counts = Cn(y)
can_stratify = all(c >= 2 for c in counts.values())
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
)
print("Distribución y_train:", Cn(y_train))
print("Distribución y_test :", Cn(y_test))

# 3) Vectorizadores 
vectorizers = {
    "BoW": CountVectorizer(max_features=10000, ngram_range=(1,2),stop_words=stopwords),
    "TFIDF": TfidfVectorizer(max_features=10000, ngram_range=(1,2),stop_words=stopwords),
}

# 4) Experimentos (modelos + HPs)

experiments = {
    "Naive Bayes": [
        MultinomialNB(alpha=1.0),
        MultinomialNB(alpha=0.1),
        MultinomialNB(alpha=0.01),
    ],
    "Logistic Regression": [
        LogisticRegression(max_iter=100,  class_weight="balanced", multi_class="ovr"),
        LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="ovr"),
    ],
    "SVM": [
        LinearSVC(C=1.0, class_weight="balanced"),
        LinearSVC(C=2.0, class_weight="balanced"),
    ],
}


# 5) Baseline (clase mayoritaria)

from collections import Counter
maj = Counter(y_train).most_common(1)[0][0]
y_base = [maj] * len(y_test)
print("\n=== Baseline (clase mayoritaria) ===")
print(classification_report(y_test, y_base, digits=3))


# 6) Entrenamos y evaluamos todo

rows = []

for vec_name, vec in vectorizers.items():
    print(f"\n\n>>> Vectorizador: {vec_name}")
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    for family, models in experiments.items():
        for m in models:
            name = f"{m.__class__.__name__} ({vec_name})"
            print(f"\nEntrenando {name} ...")
            m.fit(Xtr, y_train)
            y_pred = m.predict(Xte)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
            f1w = f1_score(y_test, y_pred, average="weighted")

            print(f"Accuracy: {acc:.3f} | Macro-F1: {f1m:.3f} | Weighted-F1: {f1w:.3f}")
            print("Reporte de clasificación:\n", classification_report(y_test, y_pred, digits=3))
            print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

            rows.append({
                "Modelo": m.__class__.__name__,
                "Vectorización": vec_name,
                "Hiperparámetros": {k: v for k, v in m.get_params().items() if k in ("alpha","C","max_iter","class_weight","multi_class")},
                "Accuracy": acc,
                "Macro-F1": f1m,
                "Weighted-F1": f1w,
            })


# 7) Tabla comparativa final

resultados = pd.DataFrame(rows).sort_values(by=["Macro-F1","Weighted-F1"], ascending=False)
print("\n\n=== Comparación de modelos (ordenado por Macro-F1) ===")
print(resultados[["Modelo","Vectorización","Hiperparámetros","Accuracy","Macro-F1","Weighted-F1"]].to_string(index=False))

