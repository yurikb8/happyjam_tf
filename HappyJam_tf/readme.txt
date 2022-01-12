学習データの取得方法
1. 学習データを”静止モーション”、”演奏モーション”で分けてトリミングしてから実行してください。
2. 動画データをmovieフォルダに
3. getLearningData_mp.pyの13行目(--gesture_id)のdefaultをモーションによって対応する番号に変更
4. getLearningData_mp.pyの14行目(--time)のdefaultを動画時間(sec)によって書き換える
5. getLearningData_mp.pyの51行目(filepath)を対応する動画のパスに書き換える
6. getLearningData_mp.pyを実行
7. npm.npzが生成されていれば成功
※73行目～77行目のコメントアウトを外せば、ポーン情報の表示が可能

TensorFlowによる学習実行方法
1. すべてのラベルについて、十分な学習データを集めてから実行しないとエラー吐くので注意
2. HappyJam_tf.pyを実行
3. tfjs_model, saved_model, gesture_classfier.tfliteの3種のファイルが生成されていれば成功


