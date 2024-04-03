import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.interpolated()
        self.abmormaled()
        self.result=self.fill_missing_values()
    # ... (다른 클래스 함수 정의는 여기에 있어야 합니다)

    def interpolated(self):
        temp = []
        for i in self.df['ID'].unique():
            temp.append(self.df[self.df['ID'] == i].interpolate())
        new_df = pd.concat(temp, axis=0)
        new_df.fillna(method='bfill', inplace=True)
        print(1)
        self.df = new_df

    def abmormaled(self):
        temp = []
        for i in self.df['ID'].unique():
            temp_df = self.df[self.df['ID'] == i].reset_index(drop=True)
            for i in range(1, len(temp_df)):
                if temp_df.at(i, 'height') <= temp_df.at(i - 1, 'height'):
                    temp_df.at[i, 'height'] = temp_df.at(i - 1, 'height')

                if self.gap(temp_df.at(i - 1, 'height'), temp_df.at(i, 'height')) >= 16:
                    temp_df.at[i, 'height'] = temp_df.at(i - 1, 'height')
            temp.append(temp_df)
        new_df2 = pd.concat(temp, axis=0)
        new_df2.reset_index(drop=True, inplace=True)
        self.df = new_df2

    def fill_missing_values(self):
        self.df["step_count"].fillna(self.df["step_count"].median(), inplace=True)
        self.df["burned calory"].fillna(self.df["burned calory"].median(), inplace=True)
        self.df["eat_calory"].fillna(self.df["eat_calory"].median(), inplace=True)
        self.df["Sleep_time"].fillna(self.df["Sleep_time"].median(), inplace=True)
        return self.df

    def save_to_csv(self, filename):
        self.df.to_csv(filename, encoding="utf-8", index=False)

# 데이터프레임 생성 및 클래스 인스턴스 생성
# df = pd.read_csv("your_data.csv")  # 데이터를 로드하는 코드
# processor = DataProcessor(df)


# if __name__ == "__main__":
#     def process_dataframe(df):
#         # 클래스 초기화
#         data_processor = DataProcessor(df)
#
#         # 데이터 처리 메서드 호출
#         data_processor.interpolated()
#         data_processor.abmormaled()
#         data_processor.fill_missing_values()
#
#         # 결과 데이터프레임 저장
#         data_processor.save_to_csv("Final_dataset.csv")
#         return(data_processor)
#         # 다른 주피터 파일에서 데이터프레임을 읽고 전달
#     df = pd.read_csv("./merged_df_reward.csv",encoding="utf-8")
#     return_df=process_dataframe(df)







