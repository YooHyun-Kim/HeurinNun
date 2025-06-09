package fckh.docgrading.controller;

import fckh.docgrading.api.LoginDto;
import fckh.docgrading.domain.Member;
import fckh.docgrading.service.LogService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Slf4j
@Controller
@RequiredArgsConstructor
public class LogController {

    private final LogService loginService;

    @GetMapping("/login")
    public String loginForm(Model model) {
        model.addAttribute("loginDto", new LoginDto());
        return "login";
    }

    @PostMapping("/login")
    public String login(@ModelAttribute LoginDto loginDto, BindingResult result, HttpServletRequest request) {
        if (result.hasErrors()) {
            return "login";
        }

        Member loginMember = loginService.login(loginDto.getLoginId(), loginDto.getPassword());
        if (loginMember == null) {
            result.reject("loginFail","아이디 또는 비밀번호가 맞지 않습니다.");
            return "login";
        }

        HttpSession session = request.getSession();
        session.setAttribute("loginMember", loginMember);
        return "redirect:/";
    }

    @RequestMapping("/logout")
    public String logout(HttpServletRequest request) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            session.invalidate();
        }
        return "redirect:/";
    }

}
