package fckh.docgrading.api;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class MemberDto {

    private String name;
    private String email;
    private String userId;
    private String password;

    public MemberDto(String userId, String password) {
        this.userId = userId;
        this.password = password;
    }
}
