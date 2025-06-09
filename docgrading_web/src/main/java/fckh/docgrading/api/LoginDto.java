package fckh.docgrading.api;

import jakarta.validation.constraints.NotEmpty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class LoginDto {

    @NotEmpty(message = "아이디를 입력하세요.")
    private String loginId;

    @NotEmpty(message = "패스워드를 입력하세요.")
    private String password;
}
