package fckh.docgrading.controller;

import fckh.docgrading.api.MemberDto;
import fckh.docgrading.domain.Member;
import fckh.docgrading.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
@RequiredArgsConstructor
public class SignupController {

    private final MemberService memberService;

    @GetMapping("/signup")
    public String signupForm(Model model) {
        model.addAttribute("memberDto", new MemberDto());
        return "signup";
    }

    @PostMapping("/signup")
    public String signup(MemberDto memberDto, BindingResult result) {
        if (result.hasErrors()) {
            return "signup";
        }
        Member member
                = new Member(memberDto.getName(), memberDto.getEmail(), memberDto.getUserId(), memberDto.getPassword());
        memberService.addMember(member);

        return "redirect:/login";
    }
}
