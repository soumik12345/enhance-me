from PIL import Image
import streamlit as st
from tensorflow.keras import utils

from enhance_me.mirnet import MIRNet


@st.cache
def get_mirnet_object() -> MIRNet:
    mirnet = MIRNet()
    mirnet.build_model()
    utils.get_file(
        "weights_lol_128.h5",
        "https://github.com/soumik12345/enhance-me/releases/download/v0.2/weights_lol_128.h5",
        cache_dir=".",
        cache_subdir="weights",
    )
    mirnet.load_weights("./weights/weights_lol_128.h5")
    return mirnet


def main():
    st.markdown("# Enhance Me")
    st.markdown("Made with :heart: by [geekyRakshit](http://github.com/soumik12345)")
    application = st.sidebar.selectbox(
        "Please select the application:", ("", "Low-light enhancement")
    )
    if application != "":
        if application == "Low-light enhancement":
            uploaded_file = st.sidebar.file_uploader("Select your image:")
            if uploaded_file is not None:
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="original image")
                mirnet = get_mirnet_object()
                enhanced_image = mirnet.infer(original_image)
                st.image(enhanced_image, caption="enhanced image")


if __name__ == "__main__":
    main()
