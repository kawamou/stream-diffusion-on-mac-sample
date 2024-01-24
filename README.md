# stream-diffusion-on-mac-sample

StreamDiffusion を macOS で動かすサンプルです。

## 実行方法

```sh
poetry shell
poetry install
```

`.venv/lib/python3.XX/site-packages/streamdiffusion`以下の`pipeline.py`の`cuda`依存部分を修正。

<img width="1075" alt="image" src="https://github.com/kawamou/stream-diffusion-on-mac-sample/assets/18514782/c60c7252-0076-4a49-bcff-a932f2e04bdd">

修正後実行。

```sh
python -m app.examples.web-camera
```
