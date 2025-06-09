package fckh.docgrading.domain;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Document {

    @Id
    @GeneratedValue
    @Column(name = "doc_id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "member_id")
    private Member member;

    @Embedded
    private UploadFile attachFile;

    private LocalDateTime uploadDate;
    private String grade;

    @Column(columnDefinition = "LONGTEXT")
    private String reason;

    private List<String> keywords = new ArrayList<>();
    private List<Integer> sampledPages = new ArrayList<>();
    private List<Integer> gradeOnePages = new ArrayList<>();
    private List<Integer> gradeTwoPages = new ArrayList<>();
    private List<Integer> gradeThreePages = new ArrayList<>();

    public static Document createDocument(Member member, UploadFile file, String grade, String reason, List<String> keywords,
                                          List<Integer> sampledPages, List<Integer> gradeOnePages, List<Integer> gradeTwoPages, List<Integer> gradeThreePages) {
        Document document = new Document();
        document.attachFile = file;
        document.uploadDate = LocalDateTime.now();
        document.grade = grade;
        document.reason = reason;
        document.member = member;

        document.keywords = keywords;
        document.sampledPages = sampledPages;
        document.gradeOnePages = gradeOnePages;
        document.gradeTwoPages = gradeTwoPages;
        document.gradeThreePages = gradeThreePages;

        member.getDocuments().add(document);
        return document;
    }

}
