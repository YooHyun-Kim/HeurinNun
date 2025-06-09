package fckh.docgrading.domain;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Member {

    @Id
    @GeneratedValue
    @Column(name = "member_id")
    private Long id;

    @OneToMany(mappedBy = "member", cascade = CascadeType.ALL)
    private List<Document> documents = new ArrayList<>();

    private String name;
    private String email;
    private String userId;
    private String password;

    public Member(String name, String email, String userId, String password) {
        this.name = name;
        this.email = email;
        this.userId = userId;
        this.password = password;
    }
}
