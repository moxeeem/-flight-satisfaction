import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

DATASET_PATH = "clients.csv"
BEST_MODEL_PATH = "logistic_regression_model.pkl"

st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Удовлетворенность полетом",
        page_icon=":airplane:",
    )

image = Image.open('img/plane.jpg')

st.write(
        """
        # Прогнозирование удовлетворенности пассажиров авиакомпании
        Определяем, кто из пассажиров доволен полетом, а кто - нет.
        """
    )

st.image(image)

st.sidebar.header('Введите параметры')

def sidebar_input_features():
    age = st.sidebar.slider('Ваш возраст: ', 7, 114, 4, key="0")

    customer_choice = st.sidebar.radio('Тип клиента: ', ('Лоялен компании', 'Нелоялен компании'), key="1")
    if customer_choice == 'Лоялен компании':
        customer_type = 0
    else:
        customer_type = 1

    customer_choice = st.sidebar.radio('Тип полета: ', ('Деловой перелет', 'Личный перелет'), key="2")
    if customer_choice == 'Деловой перелет':
        type_of_travel = 0
    else:
        type_of_travel = 1

    options = ['Eco', 'Eco Plus', 'Business']
    default_index = 2
    customer_choice = st.sidebar.radio('Класс обслуживания: ', options, index=default_index, key="3")
    if customer_choice == options[1]:
        cclass = 1
    elif customer_choice == options[0]:
        cclass = 0
    else:
        cclass = 2

    flight_distance = st.sidebar.slider('Дальность полета (в милях): ', 5, 8000, 500, key="4")
    departure_delay_in_minutes = st.sidebar.slider('Задержка отправления (в минутах): ', 5, 11500, 100, key="5")
    inflight_wifi_service = st.sidebar.selectbox('Качество интернета на борту: ', [1, 2, 3, 4, 5], key="6")
    ease_of_online_booking = st.sidebar.selectbox('Удобство онлайн-бронирования: ', [1, 2, 3, 4, 5], key="7")
    food_and_drink = st.sidebar.selectbox('Оценка еды и напитков на борту: ', [1, 2, 3, 4, 5], key="8")
    online_boarding = st.sidebar.selectbox('Удобство онлайн выбора места: ', [1, 2, 3, 4, 5], key="9")
    seat_comfort = st.sidebar.selectbox('Удобство сиденья: ', [1, 2, 3, 4, 5], key="10")
    inflight_entertainment = st.sidebar.selectbox('Оценка развлечений на борту: ', [1, 2, 3, 4, 5], key="11")
    on_board_service = st.sidebar.selectbox('Оценка питания (обслуживание): ', [1, 2, 3, 4, 5], key="12")
    leg_room_service = st.sidebar.selectbox('Оценка места в ногах на борту: ', [1, 2, 3, 4, 5], key="13")
    baggage_handling = st.sidebar.selectbox('Оценка обращения с багажом: ', [1, 2, 3, 4, 5], key="14")
    checkin_service = st.sidebar.selectbox('Удобство регистрации: ', [1, 2, 3, 4, 5], key="15")
    inflight_service = st.sidebar.selectbox('Обслуживание:', [1, 2, 3, 4, 5], key="16")
    cleanliness = st.sidebar.selectbox('Чистота: ', [1, 2, 3, 4, 5], key="17")

    data = {'Age': age,
            'Customer Type': customer_type,
            'Type of Travel': type_of_travel,
            'Class': cclass,
            'Flight Distance': flight_distance,
            'Departure Delay in Minutes': departure_delay_in_minutes,
            'Inflight wifi service': inflight_wifi_service,
            'Ease of Online booking': ease_of_online_booking,
            'Food and drink': food_and_drink,
            'Online boarding': online_boarding,
            'Seat comfort': seat_comfort,
            'Inflight entertainment': inflight_entertainment,
            'On-board service': on_board_service,
            'Leg room service': leg_room_service,
            'Baggage handling': baggage_handling,
            'Checkin service': checkin_service,
            'Inflight service': inflight_service,
            'Cleanliness': cleanliness,
            }

    df = pd.DataFrame(data, index=[0])
    return df

df = sidebar_input_features()

st.subheader('Введенные параметры')
st.write(df)

dataset = pd.read_csv('clients.csv')
X = dataset.drop(['satisfaction'], axis=1)
y = dataset['satisfaction']

logistic_para = {
        'C': 0.01,
        'penalty': 'l2',
        'random_state': 0,
        'max_iter': 100
    }

model = LogisticRegression(**logistic_para)

model.fit(X, y)
probability = model.predict_proba(df)


st.subheader('Предсказание')
st.write(f'Пассажир останется удовлетворен с вероятностью {probability[:,1].round(4).item()}')