# Standing on the Shoulders of Giants: 🤗 Transformers

In the previous lab, you learnt how to build GPT2 - but we didn't cover anything beyond the architecture.
This will change today, as you get to play with pretrained transformers!

We are going to build the one and only true demo application for a pretrained transformer: a chatbot. You will be using the Hugging Face `transformers` library and `streamlit`, a library to create web apps for machine learning and data science.

To bring you up to speed with `transformers`, we have prepared a notebook for you. It will cover the basics of downloading and interacting with pretrained transformers, as well as some foundational prompt engineering: [transformers.ipynb](notebooks/transformers.ipynb) ([solution](notebooks/transformers_solution.ipynb)).

Afterwards, you will learn how to build beautiful UIs for you data science application using `streamlit`!

| Part | Topic | Link |
|:---:|:---|:---|
| 2a | LLMs with `transformers` | [transformers.ipynb](notebooks/transformers.ipynb) ([solution](notebooks/transformers_solution.ipynb)) |
| 2b | UIs with `streamlit` | [streamlit_uis.md](./streamlit_uis.md) |
| 2c | Chatbots with `streamlit` | [streamlit_chatbot.md](./streamlit_chatbot.md) |

## Further reading

Both on the topic of LLMs and UIs for data visualization there are many more alternatives and extensions to what you've seen in this lab.
If you want to learn more about prompt engineering and building LLM-powered applications, consider taking a closer look at one of the following:

- [`langchain`](https://python.langchain.com/docs/get_started/introduction), a framework for developing context-aware applications powered by language models.
- [`llama-index`](https://docs.llamaindex.ai/en/stable/), another library for building "retrieval-augmented generation" (context-aware LLM applications).
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp) / [`ggml`](https://github.com/ggerganov/ggml) for `C/C++` implementations of Llama and many other big models - in case you are more interested in the "low-level" details.

If you are interested in data visualization, here are a few pointers to libraries and resources that could be interesting.

- [`from Data to Viz`](https://www.data-to-viz.com/), a website that helps you choose the right visualization method for your data.
- [`Plotly`](https://plotly.com/python/), a "graphing library makes interactive, publication-quality graphs"; an alternative to `matplotlib`.
- [`Plotly Dash`](https://dash.plotly.com/tutorial), the plotly-based alternative to `streamlit`.
- [`Streamsync`](https://www.streamsync.cloud/), a lightning-fast alternative to Streamlit.
