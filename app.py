import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
from sklearn.manifold import TSNE
import plotly.graph_objects as go



MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'

DATA = 'welcome_survey_simple_v1.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

@st.cache_data
def generate_tsne_visualization(all_df, person_df):
    # Przygotuj dane - najpierw kopiujemy, żeby nie modyfikować oryginalnych danych
    all_df_copy = all_df.copy()
    person_df_copy = person_df.copy()
    
    # Konwertujemy kategoryczne wartości wieku na wartości liczbowe
    age_mapping = {
        '<18': 1,
        '18-24': 2,
        '25-34': 3,
        '35-44': 4,
        '45-54': 5,
        '55-64': 6,
        '>=65': 7,
        'unknown': 0
    }
    
    # Zastosuj mapowanie do kolumny 'age'
    all_df_copy['age_numeric'] = all_df_copy['age'].map(age_mapping)
    person_df_copy['age_numeric'] = person_df_copy['age'].map(age_mapping)
    
    # Usuwamy kolumnę 'Cluster' z person_df jeśli istnieje, aby uniknąć konfliktów
    if 'Cluster' in person_df_copy.columns:
        person_df_copy = person_df_copy.drop('Cluster', axis=1)
    
    # Dodajemy tymczasową kolumnę 'Cluster' do person_df
    if 'Cluster' in all_df_copy.columns:
        # Używamy wartości klastra z przewidywania
        person_df_copy['Cluster'] = predict_model(model, data=person_df_copy)["Cluster"].values[0]
    else:
        # Jeśli all_df nie ma kolumny Cluster, używamy wartości -1 dla osoby
        person_df_copy['Cluster'] = -1
    
    # Łączymy dane
    combined_df = pd.concat([all_df_copy, person_df_copy], ignore_index=True)
    
    # Tworzymy flagę wskazującą, który wiersz to użytkownik
    combined_df['is_user'] = [False] * len(all_df_copy) + [True] * len(person_df_copy)
    
    # Wybieramy tylko kolumny kategoryczne do kodowania
    categorical_cols = ['edu_level', 'fav_animals', 'fav_place', 'gender']
    
    # Wykonujemy one-hot encoding na kolumnach kategorycznych
    combined_encoded = pd.get_dummies(combined_df, columns=categorical_cols)
    
    # Wybieramy tylko kolumny liczbowe dla t-SNE - ważne, żeby usunąć oryginalne kolumny z danymi tekstowymi
    numeric_columns = ['age_numeric'] + [col for col in combined_encoded.columns 
                                        if col.startswith(('edu_level_', 'fav_animals_', 'fav_place_', 'gender_'))]
    
    # Tworzymy DataFrame tylko z kolumnami liczbowymi
    numeric_data = combined_encoded[numeric_columns]
    
    # Sprawdzamy, czy mamy jakieś kolumny z wartościami pustymi
    for col in numeric_data.columns:
        if numeric_data[col].isna().any():
            numeric_data[col] = numeric_data[col].fillna(0)
    
    # Wykonaj redukcję wymiarowości z t-SNE
    # Dostosujemy perplexity, aby nie było większe niż liczba próbek minus 1
    perplexity_value = min(30, len(numeric_data) - 1)
    if perplexity_value < 5:  # t-SNE potrzebuje sensownej wartości perplexity
        perplexity_value = 5
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, learning_rate='auto')
    tsne_result = tsne.fit_transform(numeric_data.values)
    
    # Przygotuj dane do wizualizacji
    tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
    tsne_df['Cluster'] = combined_df['Cluster'].astype(str)
    tsne_df['is_user'] = combined_df['is_user']
    
    return tsne_df

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

#Wizualizacja ozkładu cech z pomocą t-SNE
#Wizualizacja pokaże użytkownikowi, jak plasuje się on w przestrzeni cech 
# w porównaniu do innych osób
st.header("Twoja pozycja wśród innych uczestników")
tsne_df = generate_tsne_visualization(all_df, person_df)

# Przygotuj kolory dla klastrów
colors = px.colors.qualitative.Plotly
color_map = {str(i): colors[i % len(colors)] for i in range(len(cluster_names_and_descriptions))}

# Sprawdź format cluster_id w danych
sample_cluster_id = tsne_df['Cluster'].iloc[0] if not tsne_df.empty else "0"
has_prefix = sample_cluster_id.startswith("Cluster ")

# Dostosuj color_map jeśli potrzebne
if has_prefix:
    # Jeśli ID klastra ma format "Cluster X", dostosuj mapę kolorów
    color_map = {f"Cluster {i}": colors[i % len(colors)] for i in range(len(cluster_names_and_descriptions))}

# Utwórz wizualizację Plotly
fig = go.Figure()

# Dodaj punkty dla każdego klastra
for cluster_id in tsne_df['Cluster'].unique():
    cluster_data = tsne_df[tsne_df['Cluster'] == cluster_id]
    regular_points = cluster_data[~cluster_data['is_user']]
    
    if len(regular_points) > 0:
        # Pobierz nazwę klastra lub użyj samego ID jeśli nie ma opisu
        cluster_name = cluster_id
        
        # Sprawdź, czy musimy usunąć prefiks "Cluster " do znalezienia ID w słowniku
        lookup_id = cluster_id
        if has_prefix and not cluster_id.startswith("Cluster "):
            lookup_id = f"Cluster {cluster_id}"
        elif not has_prefix and cluster_id.startswith("Cluster "):
            lookup_id = cluster_id.replace("Cluster ", "")
        
        # Spróbuj znaleźć nazwę klastra w opisach
        try:
            if lookup_id in cluster_names_and_descriptions:
                cluster_name = cluster_names_and_descriptions[lookup_id]['name']
            elif str(lookup_id) in cluster_names_and_descriptions:
                cluster_name = cluster_names_and_descriptions[str(lookup_id)]['name']
        except:
            # Jeśli nie znaleziono nazwy, użyj ID jako nazwy
            pass

        # Wybierz kolor dla klastra
        try:
            cluster_color = color_map[lookup_id]
        except KeyError:
            # Jeśli nie znaleziono koloru, użyj domyślnego
            cluster_color = colors[int(cluster_id) % len(colors) if cluster_id.isdigit() else 0]
        
        fig.add_trace(go.Scatter(
            x=regular_points['x'], 
            y=regular_points['y'],
            mode='markers',
            marker=dict(color=cluster_color, size=8),
            name=f"Grupa {cluster_name}"
        ))

# Dodaj punkt użytkownika
user_point = tsne_df[tsne_df['is_user']]
if not user_point.empty:
    fig.add_trace(go.Scatter(
        x=user_point['x'], 
        y=user_point['y'],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name="Ty"
    ))

fig.update_layout(
    title="Mapa uczestników - zobacz, gdzie pasujesz",
    xaxis_title="Wymiar 1",
    yaxis_title="Wymiar 2",
    legend_title="Grupy"
)

st.plotly_chart(fig)

