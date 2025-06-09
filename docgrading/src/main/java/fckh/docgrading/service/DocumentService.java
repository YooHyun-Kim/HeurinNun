package fckh.docgrading.service;

import fckh.docgrading.domain.Document;
import fckh.docgrading.domain.UploadFile;
import fckh.docgrading.repository.DocumentRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class DocumentService {

    @Value("${file.dir}")
    private String fileDir;

    private final DocumentRepository documentRepository;

    @Transactional
    public void saveDocument(Document document) {
        documentRepository.save(document);
    }

    public Document findDocument(Long docId) {
        return documentRepository.findById(docId).orElse(null);
    }

    public List<Document> findDocuments() {
        return documentRepository.findAll();
    }

    public UploadFile storeFile(MultipartFile file) throws IOException {

        log.info("file = {}", file);
        if (file.isEmpty()) {
            return null;
        }

        String originalFilename = file.getOriginalFilename();
        String storeFileName = createStoreFileName(originalFilename);

        String fullPath = getFullPath(storeFileName);
        log.info("full path = {}", fullPath);
        file.transferTo(new File(fullPath));

        return new UploadFile(originalFilename, storeFileName);
    }

    public String getFullPath(String fileName) {
        return fileDir + fileName;
    }

    // 서버 내부에서 관리하는 파일명은 유일한 이름을 생성하는 UUID 를 사용
    private String createStoreFileName(String originalFilename) {
        String ext = extractExt(originalFilename);
        String uuid = UUID.randomUUID().toString();
        return uuid + "." + ext;
    }
    // 확장자를 별도로 추출해서 서버 내부에서 관리하는 파일명에도 붙여줌
    private String extractExt(String originalFilename) {
        int pos = originalFilename.lastIndexOf(".");
        return originalFilename.substring(pos + 1);
    }
}
