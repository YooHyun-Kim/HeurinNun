package fckh.docgrading.repository;

import fckh.docgrading.domain.Member;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface MemberRepository extends JpaRepository<Member, Long> {

    @Query("select m from Member m where m.userId = :loginId")
    Optional<Member> findByLoginId(String loginId);

    @Query("select m from Member m join fetch m.documents where m.userId = :loginId") // 추가
    Optional<Member> findByLoginIdWithDocuments(String loginId); // 추가

}
