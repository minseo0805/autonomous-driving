from flask import Flask, request, render_template, jsonify
import os
import psycopg2
import pandas as pd
import time
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
import joblib  # 'joblib' 모듈 가져오기


app = Flask(__name__)

# TimescaleDB 연결 설정
DB_HOST = 'localhost'
DB_PORT = 5432
DB_USER = 'postgres'
DB_PASSWORD = 'qwer1234!!'
DB_NAME = 'car'

# 업로드된 파일을 저장할 디렉토리 설정
UPLOAD_FOLDER = 'C:\\Users\\minseo\\AppData\\Local\\Programs\\Python\\Python311\\Scripts\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 모델 파일 경로 설정
model_filename = 'C:\\Users\\minseo\\AppData\\Local\\Programs\\Python\\Python311\\Scripts\\RF_3.joblib'

# 모델을 로드
loaded_model = joblib.load(model_filename)

# 모델을 로드한 이후에 n_features_in_ 속성 확인
n_features = loaded_model.n_features_in_

# 특성 수 출력
print(f"모델의 특성 수: {n_features}")


# TimescaleDB 연결 함수
def connect_to_db():
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME)
    return conn

# CSV 파일을 TimescaleDB에 삽입
def insert_csv_data_to_db(file_path):
    conn = connect_to_db()
    cursor = conn.cursor()

    # CSV 파일을 데이터베이스 테이블에 삽입
    with open(file_path, 'r') as f:
        cursor.copy_expert("COPY timeseries_data FROM STDIN WITH CSV HEADER", f)

    conn.commit()
    conn.close()

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        start_time = time.time()

        # 데이터를 쿼리하여 Pandas DataFrame으로 가져오기
        query = """
                SELECT 
                egoVehicleSpeedMps,
                steeringAngleDeg,
                steeringtorque,
                ttcOfFrontVehicleS,
                distanceOfFrontVehicleM,
                velocityOfFrontVehicleMps
                FROM timeseries_data
                """
        cursor.execute(query)
        data = cursor.fetchall()

        if data:
            # 데이터를 Pandas DataFrame으로 변환
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)

            # 윈도우 사이즈 설정
            window_size = 10  # 10개의 데이터 포인트로 윈도우 생성

            # 데이터 배열 생성 및 윈도우로 변환
            data_arrays = []
            for i in range(len(df) - window_size + 1):
                sub_df = df.iloc[i:i + window_size]
                sub_df = sub_df.transpose()  # 행 열 전환

                # DataFrame를 배열로 변환
                sub_array = sub_df.values.flatten().tolist()
                data_arrays.append(sub_array)

            # data_arrays를 NumPy 배열로 변환
            X_test = np.array(data_arrays)

            # 예측 수행
            y_pred_rf = predict_with_expanded_data(X_test)

            end_time = time.time()
            fitting_time_ms = (end_time - start_time) * 1000
            print(fitting_time_ms)

            # 예측 결과를 기반으로 분석 판단 시스템을 적용
            analysis_result = []

            for prediction in y_pred_rf:
                # 각 항목에 대한 분석을 수행
                item_analysis_result = []

                for item in prediction:
                    if 0.0 <= item <= 0.5:
                        item_analysis_result.append("보통")
                    elif 0.6 <= item <= 0.8:
                        item_analysis_result.append("경고")
                    elif item > 0.8:
                        item_analysis_result.append("위험")
                    else:
                        item_analysis_result.append("보통")  # 다른 경우, 분석 중으로 표시

                analysis_result.append(item_analysis_result)

            # 분석 결과 반환
            return jsonify({'analysis_result': analysis_result})

        else:
            return jsonify({'error': '데이터를 찾을 수 없습니다.'}), 404

    except Exception as e:
        # 예외 처리와 함께 오류 메시지를 로깅
        app.logger.error(f"An error occurred in /get_predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# 데이터를 반복하여 feature 수를 720으로 확장하는 함수
def expand_data_to_720_features(data):
    current_features = len(data[0])

    if current_features >= 720:
        return data  # 이미 720개 이상의 특성을 갖고 있다면 수정하지 않고 그대로 반환

    remaining_features = 720 - current_features
    expanded_data = []

    for point in data:
        expanded_point = np.concatenate([point, np.zeros(remaining_features)])  # 부족한 특성을 0으로 채움
        expanded_data.append(expanded_point)

    return expanded_data


 
# 예측 시에 데이터를 확장하여 720개의 feature를 갖도록 함
def predict_with_expanded_data(data):
    # start_time = time.time()
    # 데이터 확장
    expanded_data = expand_data_to_720_features(data)
    # end_time = time.time()
    # fitting_time_ms = (end_time - start_time) * 1000
    # print(fitting_time_ms)

    # 예측 수행
    y_pred_rf = loaded_model.predict_proba(expanded_data)

    return y_pred_rf   


            
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # 클라이언트로부터 업로드된 파일을 받음
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # 업로드된 파일의 원래 이름을 사용하여 파일 경로를 구성
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)

            # 파일을 업로드할 디렉토리에 저장
            uploaded_file.save(file_path)

            # CSV 파일을 TimescaleDB에 삽입
            insert_csv_data_to_db(file_path)

            # 파일 업로드가 성공하면 간단한 메시지 반환
            return "파일 업로드가 완료되었습니다."

    return render_template("upload.html")  # 업로드 폼을 보여주는 HTML 페이지

@app.route("/upload", methods=["POST"])
def handle_upload():
    if request.method == "POST":
        # 클라이언트로부터 업로드된 파일을 받음
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # 업로드된 파일의 원래 이름을 사용하여 파일 경로를 구성
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)

            # 파일을 업로드할 디렉토리에 저장
            uploaded_file.save(file_path)

            # 이전 데이터를 삭제
            delete_existing_data()

            # CSV 파일을 TimescaleDB에 삽입
            insert_csv_data_to_db(file_path)

            # 업로드된 파일을 저장한 경로를 반환
            return f"파일 업로드 및 데이터베이스에 삽입 완료. 파일 경로: {file_path}"

    return "파일 업로드 실패"

def delete_existing_data():
    conn = connect_to_db()
    cursor = conn.cursor()

    try:
        # timeseries_data 테이블의 모든 데이터 삭제
        query = "DELETE FROM timeseries_data"
        cursor.execute(query)
        conn.commit()
    except Exception as e:
        print(f"데이터 삭제 중 오류 발생: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # 데이터를 쿼리하여 Pandas DataFrame으로 가져오기
        query = "SELECT * FROM timeseries_data"
        cursor.execute(query)
        data = cursor.fetchall()

        if data:
            # 모든 데이터 행을 저장할 리스트
            result = []

            for row in data:
                columns = [desc[0] for desc in cursor.description]
                data_dict = dict(zip(columns, row))
                result.append(data_dict)

            return jsonify(result)
        else:
            return jsonify({'error': '데이터를 찾을 수 없습니다.'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



def fetch_data_from_db():
    with app.app_context():
        conn = connect_to_db()
        cursor = conn.cursor()

        try:
            query = """
            SELECT * FROM timeseries_data
            """

            cursor.execute(query)
            data = cursor.fetchall()  # 모든 데이터 행을 가져옴

            result = []  # 모든 데이터 행을 저장할 리스트

            for row in data:
                columns = [desc[0] for desc in cursor.description]
                data_dict = dict(zip(columns, row))
                result.append(data_dict)

            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()



# 스케줄러 설정
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_data_from_db, 'interval', seconds=1, max_instances=50)
scheduler.start()




@app.route("/monitoring.html")
def monitoring():
    return render_template("monitoring.html")

if __name__ == "__main__":
    app.run(debug=True)