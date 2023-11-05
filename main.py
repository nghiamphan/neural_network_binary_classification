import streamlit as st
import pandas as pd
import NeuralNetwork as NN
import utils as U

st.set_page_config(
    page_title="Simple Neural Network for Binary Classification",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Simple Neural Network for Binary Classification")


def intro():
    st.write("")
    expander = st.expander("#### Introduction", expanded=True)
    expander.write(
        """
        This is a simple neural network for binary classification I wrote from scatch. The network consists of:
        - an input layer
        - zero or one hidden layer of customizable number of nodes; activation function: leaky RELU.
        - one output layer, activation: binary cross entrophy.

        Input vector: list of numbers

        Output: 0 or 1

        Given a dataset, the model will split it into training and test set. The model will train on the training set, and then print out results on the test set.
                """
    )


def input():
    st.write("")
    expander = st.expander("#### Choose data source", expanded=True)

    uploaded_file = expander.file_uploader("Upload data file")
    header = expander.toggle("Uploaded file contains header?", value=True)
    use_sample_data = expander.toggle(
        "Use sample data from https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        disabled=uploaded_file != None,
    )

    df = None
    if uploaded_file is not None:
        header = None if not header else 1
        df = pd.read_csv(uploaded_file, header=header)
    elif use_sample_data:
        header = None
        df = pd.read_csv(NN.SAMPLE_FILE, header=header)

    if df is not None:
        # Set up default column names if not already exist
        if header == None:
            column_names = []
            for i in range(len(df.iloc[0]) - 1):
                column_names += [f"Feature_{i+1}"]
            column_names += ["Target"]
            df.columns = column_names

        data_exploration(df)

        st.write("")
        st.write("")

        expander_input = st.expander("#### Input paramters for the model", expanded=True)

        hidden_layer = expander_input.toggle("Hidden layer", value=True)
        n_hidden_nodes = 0
        if hidden_layer:
            n_hidden_nodes = expander_input.number_input(
                "Number of hidden nodes", min_value=1, max_value=1000, value=10, step=1, format="%i"
            )

        normalization = expander_input.toggle("Input normalization", value=True)

        test_size = expander_input.number_input(
            "Test ratio", min_value=0.01, max_value=0.99, value=0.2, step=0.1, format="%f"
        )

        epochs = expander_input.number_input("Epochs", min_value=1, max_value=10000, value=100, step=10, format="%i")

        if not hidden_layer:
            learning_rate = expander_input.number_input(
                "Learning rate", min_value=0.001, max_value=10.0, value=1.0, step=0.1, format="%f"
            )
            l2 = expander_input.number_input(
                "L2 regularization rate", min_value=0.0, max_value=0.1, value=0.01, step=0.01, format="%f"
            )
            list_learning_rate = [learning_rate]
            list_l2 = [l2]
        else:
            learning_rate_1 = expander_input.number_input(
                "Learning rate of weights from input to hidden layer",
                min_value=0.001,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%f",
            )
            learning_rate_2 = expander_input.number_input(
                "Learning rate of weights from hidden to output layer",
                min_value=0.001,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%f",
            )
            l2_1 = expander_input.number_input(
                "L2 regularization rate of weights from input to hidden layer",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.01,
                format="%f",
            )
            l2_2 = expander_input.number_input(
                "L2 regularization rate of weights from hidden to output layer",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.01,
                format="%f",
            )
            list_learning_rate = [learning_rate_1, learning_rate_2]
            list_l2 = [l2_1, l2_2]

        st.button(
            "Run model",
            on_click=run_model,
            args=(
                df,
                hidden_layer,
                n_hidden_nodes,
                normalization,
                test_size,
                epochs,
                list_learning_rate,
                list_l2,
            ),
            type="primary",
        )


def data_exploration(df: pd.DataFrame):
    expander = st.expander("#### Data exploration", expanded=True)

    expander.write("##### Random samples of data")
    expander.write(df.sample(n=10))

    expander.write("")
    expander.write("##### Sample Quick Info")
    expander.write(df.describe())

    expander.write("")
    expander.write("##### Features Distribution Visualization")
    expander.pyplot(U.box_plot(df.iloc[:, :-1]))

    expander.write("")
    expander.write("##### Target Sample Size")
    expander.write(df.iloc[:, -1].value_counts())


def run_model(
    df: pd.DataFrame,
    hidden_layer: int,
    n_hidden_nodes: int,
    normalization: bool,
    test_size: float,
    epochs: int,
    list_learning_rate: list[float],
    list_l2: list[float],
):
    if hidden_layer:
        nn = NN.NeuralNetworkHidden(n_hidden_nodes)
        nn.import_data(data_frame=df, test_size=test_size, normalization=normalization)
        test_log_loss, scores = nn.slow_train(
            nn.X_train,
            nn.y_train,
            epochs=epochs,
            learning_rate_1=list_learning_rate[0],
            learning_rate_2=list_learning_rate[1],
            l2_1=list_l2[0],
            l2_2=list_l2[1],
        )
    else:
        nn = NN.NeuralNetwork()
        nn.import_data(data_frame=df, test_size=test_size, normalization=normalization)
        test_log_loss, scores = nn.slow_train(
            nn.X_train,
            nn.y_train,
            epochs=epochs,
            learning_rate=list_learning_rate[0],
            l2=list_l2[0],
        )

    st.session_state["result_figure"] = nn.draw_result(scores)
    st.session_state["classical_report"] = nn.classification_report
    st.session_state["confusion_matrix"] = nn.draw_confusion_matrix()


def display_result():
    if "result_figure" in st.session_state and st.session_state["result_figure"]:
        st.write("")
        expander = st.expander("#### Results", expanded=True)

        expander.write("#### Accuracy and Log loss of train and test set over epochs")
        expander.pyplot(st.session_state["result_figure"])

        expander.write("")
        expander.write("#### Final Result Over Test Set")
        expander.write("##### Confusion Matrix")
        expander.pyplot(st.session_state["confusion_matrix"])

        expander.write("")
        expander.write("##### Classical Report")
        expander.text("." + st.session_state["classical_report"])


intro()
input()
display_result()
