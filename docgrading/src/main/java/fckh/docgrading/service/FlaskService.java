package fckh.docgrading.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import fckh.docgrading.api.DocumentDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional
public class FlaskService {

    @Value("${file.dir}")
    private String fileDir;
    private final ObjectMapper objectMapper;

    public DocumentDto getGradeAndReason(String fileName) {

        try {
            // Flask 서버에 파일 경로 전송하고 결과 받기
            String result = sendToFlask(fileName);

            // JSON 문자열을 Map으로 변환
            Map<String, Object> resultMap = objectMapper.readValue(result, new TypeReference<Map<String, Object>>(){});

            // 결과 데이터 추출
            String grade = (String) resultMap.get("grade");
            String reason = (String) resultMap.get("reason");
            List<String> keywords = (List<String>) resultMap.get("keyword");
            List<Integer> sampledPages = (List<Integer>) resultMap.get("sampled");
            List<Integer> gradeOnePages = (List<Integer>) resultMap.get("1급");
            List<Integer> gradeTwoPages = (List<Integer>) resultMap.get("2급");
            List<Integer> gradeThreePages = (List<Integer>) resultMap.get("3급");


            return new DocumentDto(grade, reason, keywords, sampledPages, gradeOnePages, gradeTwoPages, gradeThreePages);

        } catch (Exception e) {
            throw new RuntimeException("Flask 결과 처리 중 오류 발생", e);
        }
    }

    public String sendToFlask(String fileName)  {

        RestTemplate restTemplate = new RestTemplate();
        String flaskUrl = "http://localhost:5000/predict";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.TEXT_PLAIN);

        HttpEntity<String> request = new HttpEntity<>(fileDir + fileName, headers);

        try {
            // RestTemplate을 사용하여 Flask 서버에 요청
            return restTemplate.postForObject(flaskUrl, request, String.class);
        } catch (Exception e) {
            throw new RuntimeException("Flask 서버 통신 중 오류 발생", e);
        }
    }
}
