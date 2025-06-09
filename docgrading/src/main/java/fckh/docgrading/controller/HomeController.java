package fckh.docgrading.controller;

import fckh.docgrading.api.DocumentDto;
import fckh.docgrading.domain.Document;
import fckh.docgrading.domain.Member;
import fckh.docgrading.domain.UploadFile;
import fckh.docgrading.service.DocumentService;
import fckh.docgrading.service.FlaskService;
import fckh.docgrading.service.MemberService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.io.IOException;

@Slf4j
@Controller
@RequiredArgsConstructor
public class HomeController {

    private final MemberService memberService;
    private final DocumentService documentService;
    private final FlaskService flaskService;

    // 로그인 하지 않은 사용자도 홈에 접근할 수 있게 required = false
    @GetMapping("/")
    public String home(@SessionAttribute(required = false) Member loginMember, Model model) {

        if (loginMember == null) {
            return "home";
        }
        model.addAttribute("loginMember", loginMember);
        return "loginHome";
    }

    // 문서 업로드
    @PostMapping("/")
    public String uploadDocument(@RequestParam MultipartFile file, RedirectAttributes redirectAttributes) throws IOException {

        UploadFile attachFile = documentService.storeFile(file);
        if (attachFile == null) {
            return "loginHome";
        }

        redirectAttributes.addAttribute("uploadFilename", attachFile.getUploadFilename());
        redirectAttributes.addAttribute("storeFilename", attachFile.getStoreFilename());
        return "redirect:/loading";
    }

    @GetMapping("/loading")
    public String loading(@RequestParam String uploadFilename, @RequestParam String storeFilename, Model model) {

        model.addAttribute("uploadFilename", uploadFilename);
        model.addAttribute("storeFilename", storeFilename);
        return "loading";
    }

    @GetMapping("/sendToFlask")
    public String sendToFlask(@RequestParam String uploadFilename, @RequestParam String storeFilename,
                          @SessionAttribute(required = false) Member loginMember,
                          RedirectAttributes redirectAttributes) {

        //Flask
        DocumentDto documentDto = flaskService.getGradeAndReason(storeFilename);

        Member member = memberService.findOne(loginMember.getId());
        Document document = Document.createDocument(member, new UploadFile(uploadFilename, storeFilename),
                documentDto.getGrade(), documentDto.getReason(), documentDto.getKeywords(),
                documentDto.getSampledPages(), documentDto.getGradeOnePages(), documentDto.getGradeTwoPages(), documentDto.getGradeThreePages());
        documentService.saveDocument(document);

        redirectAttributes.addAttribute("docId", document.getId());
        return "redirect:/document/{docId}";
    }

}
