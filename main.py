import streamlit as st
import inflect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import string
import plotly.express as px
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from gsheetsdb import connect
nltk.download('punkt')

punctuations = string.punctuation

# get data for suburbs and regions

# Create a connection object.
conn = connect()


# # Perform SQL query on the Google Sheet.
# # Uses st.cache to only rerun when the query changes or after 10 min.
# @st.cache(ttl=600)
# def run_query(query):
#     rows = conn.execute(query, headers=1)
#     rows = rows.fetchall()
#     return rows
#
# sheet_url = st.secrets["https://drive.google.com/file/d/1Pyfa_pylLCB4UhDUhcy-5GBcpc3ALA1T/view?usp=sharing"]
# rows = run_query(f'SELECT * FROM "{sheet_url}"')


def prep_text(text):
    # function for preprocessing text

    # remove trailing characters (\s\n) and convert to lowercase
    clean_sents = [] # append clean con sentences
    sent_tokens = sent_tokenize(str(text))
    for sent_token in sent_tokens:
        word_tokens = [str(word_token).strip().lower() for word_token in sent_token.split()]
        word_tokens = [word_token for word_token in word_tokens if word_token not in punctuations]
        clean_sents.append(' '.join((word_tokens)))
    joined = ' '.join(clean_sents).strip(' ')

    return joined


p = inflect.engine()

# model name of path to model
checkpoint = "sadickam/vba-distilbert"


@st.cache(allow_output_mutation=True)
def load_model():
    return AutoModelForSequenceClassification.from_pretrained(checkpoint)


@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return AutoTokenizer.from_pretrained(checkpoint)


st.set_page_config(
    page_title="Domestic Building Cost", layout='wide', initial_sidebar_state="auto", page_icon="🏘️"
)

st.title("🏡 Domestic Building Construction Cost Predictor")
# st.header("")

with st.expander("About this app", expanded=False):
    st.write(
        """
        - This app was developed using basic domestic building information from the Victorian Building Authority's building permit activity data from 2020 to 2022.  
        - This app is intended for predicting the construction cost range of domestic buildings (1 to 3 stories) at the design/planning stage. The predicted cost range excludes land cost.
        - Please note that, we do not guarantee the reliability of the predicted cost range. If you need a reliable cost estimate, we will suggest that you consult a Quantity Surveyor.
        """
    )

# Project Location and floor area
b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 1.5])
with b1:
    Suburb = st.text_input(
        "Project Suburb", help="Victorian suburbs only. Suburbs outside Victoria will return wrong predictions")
with b2:
    Municipality = st.selectbox(
        "Project Municipality",
        ('Alpine', 'Ararat', 'Ballarat', 'Banyule', 'Bass Coast', 'Baw Baw', 'Bayside', 'Benalla', 'Boroondara',
         'Brimbank', 'Buloke', 'Campaspe', 'Cardinia', 'Casey', 'Central Goldfield', 'Central Goldfields',
         'Colac-Otway', 'Corangamite', 'Darebin', 'East Gippsland', 'Frankston', 'Gannawarra', 'Glen Eira',
         'Glenelg', 'Golden Plains', 'Greater Bendigo', 'Greater Dandenong', 'Greater Geelong', 'Greater Shepparton',
         'Hepburn', 'Hindmarsh', 'Hobsons Bay', 'Horsham', 'Hume', 'Indigo', 'Kingston', 'Knox', 'Latrobe', 'Loddon',
         'Macedon Ranges', 'Manningham', 'Mansfield', 'Maribyrnong', 'Maroondah', 'Melbourne', 'Melton', 'Mildura',
         'Mitchell', 'Moira', 'Monash', 'Moonee Valley', 'Moorabool', 'Moreland', 'Mornington', 'Mornington Peninsula',
         'Mount Alexander', 'Mount alexander', 'Moyne', 'Murrindindi', 'Nillumbik', 'Northern Grampians', 'Port Philip',
         'Port Phillip', 'Pyrenees', 'Queenscliff (B)', 'Queenscliffe', 'South Gippsland', 'Southern Grampians',
         'Stonnington', 'Strathbogie', 'Surf Coast', 'Swan Hill', 'Towong', 'Wangaratta', 'Warrnambool', 'Wellington',
         'West Wimmera', 'Whitehorse', 'Whittlesea', 'Wodonga', 'Wyndham', 'Yarra', 'Yarra Ranges', 'Yarriambiack'),
        help="Some Victorian Municipalities may not be listed. Select the closest if not listed"
    )
with b3:
    Region = st.selectbox("Project Region", ('Metropolitan', 'Rural'))

with b4:
    SubRegion = st.selectbox(
        "Sub-Region", ('Gippsland', 'Inner Melbourne', 'North Central', 'North East', 'North West',
                       'Outer Melbourne', 'South West')
    )

project_detail_1 = "Site is at " + Suburb + ", " + Municipality + ", " + Region + ", " + SubRegion + ". "

with b5:
    FloorArea = st.slider("Estimated total floor area", min_value=20, max_value=1000, step=10)

project_detail_3 = "Total floor area is " + p.number_to_words(FloorArea) + " square meters."

# Building materials, solar water heater and rain water storage
with st.sidebar:
    st.markdown('**Specify basic project information**')
    FloorType = st.selectbox(
        "Choose your floor type",
        ('concrete or stone', 'timber', 'other'),
        help="If your floor type is not listed, please choose 'other'"
    )

    FrameType = st.selectbox(
        "Choose your frame type",
        ('timber', 'steel', 'aluminium', 'other'),
        help="If your frame type is not listed, please choose 'other'"
    )

    RoofType = st.selectbox(
        "Choose your roof type",
        ('tiles', 'concrete or slate', 'fibre cement', 'steel', 'aluminium', 'other'),
        help="If your roof type is not listed, please choose 'other'"
    )

    WallType = st.selectbox(
        "Choose your wall type",
        ('brick double', 'brick veneer', 'concrete or stone', 'fibre cement', 'timber', 'curtain glass', 'steel',
         "Aluminium", 'Other'),
        help="If your roof type is not listed, please choose 'other'"
    )

    bldg_mat = (
            "Materials include " + FloorType + " floor, " + FrameType + " frame, " + WallType + " external wall, " +
            "and " + RoofType + " roof. ")

    Storeys = st.selectbox("Number of storey", ('one storey', 'two storey', 'three storey'))

    SolarHotWater = st.selectbox(
        "Solar hot water",
        ('Yes', 'No'),
        # help="If your roof type is not listed, please choose 'other'"
    )
    if SolarHotWater == "Yes":
        SolarHotWater = "has solar hot water"
    else:
        SolarHotWater = 'has no solar hot water'

    RainWaterTank = st.selectbox(
        "Project include rain water tank",
        ("Yes", "No")
        # help="If your roof type is not listed, please choose 'other'"
    )
    if RainWaterTank == "Yes":
        RainWaterTank = 'and rainwater tank'
    else:
        RainWaterTank = 'and no rainwater tank'

    project_detail_2 = (
            bldg_mat + "Building is " + Storeys + " and " + SolarHotWater + " " + RainWaterTank + ". ")

st.markdown("##### Project Description")
with st.form(key="my_form"):
    Project_details = st.text_area(
        "The model's prediction is based on the project description below (i.e., input). Select your options in the sidebar and above to modify the project description below",
        project_detail_1 + project_detail_2 + project_detail_3)
    submitted = st.form_submit_button(label="💵 Get cost range!")

if submitted:

    label_list = ['$300,000 and below', '$300,000 - $600,000', '$600,000 - $2M']

    joined_clean_sents = prep_text(Project_details)

    # tokenize
    tokenizer = load_tokenizer()
    tokenized_text = tokenizer(joined_clean_sents, return_tensors="pt")

    # predict
    model = load_model()
    text_logits = model(**tokenized_text).logits
    predictions = torch.softmax(text_logits, dim=1).tolist()[0]
    predictions = [round(a, 3) for a in predictions]

    # dictionary with label as key and percentage as value
    pred_dict = (dict(zip(label_list, predictions)))

    # sort 'pred_dict' by value and index the highest at [0]
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)

    # Make dataframe for plotly bar chart
    u, v = zip(*sorted_preds)
    x = list(u)
    y = list(v)
    df2 = pd.DataFrame()
    df2['Cost_Range'] = x
    df2['Likelihood'] = y

    c1, c2, c3 = st.columns([1.5, 0.5, 1])

    with c1:
        # plot graph of predictions
        fig = px.bar(df2, x="Likelihood", y="Cost_Range", orientation="h")

        fig.update_layout(
            # barmode='stack',
            template='seaborn',
            font=dict(
                family="Arial",
                size=14,
                color="black"
            ),
            autosize=False,
            width=700,
            height=300,
            xaxis_title="Likelihood of cost range",
            yaxis_title="Cost ranges",
            # legend_title="Topics"
        )

        fig.update_xaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_yaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_annotations(font_size=14)  # this changes y_axis, x_axis and subplot title font sizes

        # Plot
        st.plotly_chart(fig, use_container_width=False)

    with c3:
        st.header("")
        predicted_range = st.metric("Predicted construction cost range", sorted_preds[0][0])
        Prediction_confidence = st.metric("Prediction confidence", (str(round(sorted_preds[0][1]*100, 1))+"%"))

        st.success("Great! Cost range successfully predicted. ", icon="✅")
