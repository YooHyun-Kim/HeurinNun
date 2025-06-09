package fckh.docgrading.api;

import fckh.docgrading.domain.UploadFile;
import jakarta.persistence.Column;
import jakarta.persistence.Embedded;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class DocumentDto {

    @Embedded
    private UploadFile attachFile;
    private LocalDate uploadDate;

    private String grade;

    private List<String> keywords = new ArrayList<>();
    private List<Integer> sampledPages = new ArrayList<>();
    private List<Integer> gradeOnePages = new ArrayList<>();
    private List<Integer> gradeTwoPages = new ArrayList<>();
    private List<Integer> gradeThreePages = new ArrayList<>();

    @Column(columnDefinition = "LONGTEXT")
    private String reason;

    public DocumentDto(String grade, String reason, List<String> keywords,
                       List<Integer> sampledPages, List<Integer> gradeOnePages, List<Integer> gradeTwoPages,
                       List<Integer> gradeThreePages) {
        this.grade = grade;
        this.reason = reason;
        this.keywords = keywords;
        this.sampledPages = sampledPages;
        this.gradeOnePages = gradeOnePages;
        this.gradeTwoPages = gradeTwoPages;
        this.gradeThreePages = gradeThreePages;
    }
}
