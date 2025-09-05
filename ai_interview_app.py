            if wav_bytes:  
                with st.spinner("Transcribing audio..."):  
                    final_answer = transcribe_audio(wav_bytes)  
            if not final_answer:  
                final_answer = (answer_text_input or "").strip() or None  
            if not final_answer:  
                st.warning("No audio or typed answer found. Please record or type an answer.")  
            else:  
                with st.spinner("Evaluating answer..."):  
                    eval_obj = evaluate_answer(q, final_answer, st.session_state.get("resume",""), MODELS["GPT-4o"])  
                    eval_obj["answer"] = final_answer  
                    eval_obj.setdefault("score", 0)  
                    eval_obj.setdefault("feedback", "")  
                    st.session_state.answers.append(eval_obj)  
                    st.session_state.current_q = st.session_state.get("current_q", 0) + 1  
                    st.session_state.proctoring_img = None  
                    # Clear typed answer for next question  
                    st.session_state[f"typed_answer_{idx}"] = ""  
                    st.experimental_rerun()  
    with col_skip:  
        if st.button("Skip Question"):  
            st.session_state.current_q = st.session_state.get("current_q", 0) + 1  
            st.session_state.proctoring_img = None  
            # Clear
