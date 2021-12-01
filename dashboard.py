import predict
import config
import streamlit as st


model = predict.get_prediction_model(config.model_path)

# Title
st.title('Kollegah Raptext Generator')
st.subheader('Wie würde deinen Text vervollständigen?')

with st.form('input'):
    # Text input
    input = st.text_area('Schreibe ein paar Zeilen!', 'Ich bin Kollegah der Boss, ')
    submitted = st.form_submit_button("Generate")

output = predict.generate_text(model, input, gen_size=300)

if submitted:

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image('images/Kollegah.jpg')

    with col2:
        # Output
        st.write(output)

