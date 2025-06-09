package fckh.docgrading.service;

import fckh.docgrading.domain.Member;
import fckh.docgrading.repository.MemberRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class MemberService {

    private final MemberRepository memberRepository;

    @Transactional
    public Long addMember(Member member) {
        memberRepository.save(member);
        return member.getId();
    }

    private void validateDuplicateMember(Member member) {


    }

    public Member findOne(Long id) {
        return memberRepository.findById(id).orElse(null);
    }


}
