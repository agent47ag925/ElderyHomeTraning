import streamlit as st
import pandas as pd
import search_list
import translate
import time

st.title("Silver Lining")

keyword = st.text_input('검색 키워드를 입력하세요')
search_start = st.button('검색')

# 검색 버튼 클릭 시 DF 갱신
if keyword and search_start:
    result = search_list.search_and_get_description(keyword)

    if isinstance(result, dict):
        df = pd.DataFrame(result).T
        df.columns = ['영상 제목', '설명', '업로드 일자', '영상 길이', '영상 주소']
        df['선택'] = False
        st.session_state.df = df
    else:
        st.markdown(result)

# 세션에 DF가 있으면 항상 표시
if 'df' in st.session_state:
    edited_df = st.data_editor(
        st.session_state.df,
        key="editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "선택": st.column_config.CheckboxColumn(
                "선택",
                help="체크하면 아래에 모아서 보여줍니다."
            )
        }
    )

    st.session_state.df = edited_df

    selected_only = edited_df[edited_df["선택"] == True]
    st.subheader("체크된 항목")
    if not selected_only.empty:
	    #st.dataframe(selected_only.drop(columns=["선택"]), use_container_width=True)

	    # 영상 주소만 따로 추출
	    urls = selected_only["영상 주소"].tolist()
	    for url in urls:
	        st.markdown(f"[{url}]({url})")

	    with st.spinner("Please wait... 영상 분석 중..."):
	    	analysis_result = translate.analyze_video_with_gemini(url)
	    	st.markdown(analysis_result)

