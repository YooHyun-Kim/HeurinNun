package fckh.docgrading.controller;

import fckh.docgrading.domain.Document;
import fckh.docgrading.domain.Member;
import fckh.docgrading.service.DocumentService;
import fckh.docgrading.service.MemberService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.SessionAttribute;

import java.util.List;

@Slf4j
@Controller
@RequiredArgsConstructor
public class DocumentController {

    private final MemberService memberService;
    private final DocumentService documentService;

    // 후에 DTO로 변환
    @GetMapping("/document/{docId}")
    public String document(@PathVariable Long docId, Model model) {
        Document document = documentService.findDocument(docId);
        String fullPath = documentService.getFullPath(document.getAttachFile().getStoreFilename());

        model.addAttribute("fullPath", fullPath);
        model.addAttribute("document", document);
        return "document";
    }

    // PDF 파일을 제공하는 새로운 엔드포인트 추가
    @GetMapping("/document/{docId}/pdf")
    public ResponseEntity<Resource> viewDocumentPdf(@PathVariable Long docId) {

        try {
            Document document = documentService.findDocument(docId);
            String fullPath = documentService.getFullPath(document.getAttachFile().getStoreFilename());
            Resource resource = new FileSystemResource(fullPath);

            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_PDF)
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            "inline; filename=\"" + document.getAttachFile().getUploadFilename() + "\"")
                    .body(resource);
        } catch (Exception e) {
            log.error("PDF 파일 제공 중 오류 발생", e);
            return ResponseEntity.notFound().build();
        }
    }

    @GetMapping("/history") // 문서 목록
    public String findDocuments(@SessionAttribute(required = false) Member loginMember, Model model) {
        List<Document> documents = memberService.findOne(loginMember.getId()).getDocuments();
        model.addAttribute("loginMember", loginMember);
        model.addAttribute("documents", documents);
        return "history";
    }

}
